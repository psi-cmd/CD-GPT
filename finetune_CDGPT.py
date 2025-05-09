# -*- coding: utf-8 -*-
"""Fine‑tune CD‑GPT with **rsLoRA + Gated Adapter** (Hybrid‑PEFT)
================================================================
This version upgrades the original Lightning training script to call
`hybrid_peft_modules.apply_hybrid_peft()`, and uses
`get_peft_param_groups()` to separate the learning rates of LoRA/Adapter.
"""

from __future__ import annotations
import os
import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

from model.finetune import CDGPTFineTune, CDGPTSequenceTaskHead
from model.cd_gpt import CDGPT
from tokenizer import SentencePieceTokenizer
from DREAMdataset import load_and_process_data
from config import get_config

import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

# —— Hybrid‑PEFT util ——
from model.hybrid_peft_modules import apply_hybrid_peft, get_peft_param_groups

# -----------------------------------------------------------------------------
# LightningModule
# -----------------------------------------------------------------------------
class CDGPTLightningModule(pl.LightningModule):
    def __init__(self, cfg, has_ckpt=False):
        super().__init__()
        self.cfg = cfg

        # Build backbone and load pretrained weights (all frozen)
        self.model = CDGPT(vocab_size=cfg.tokenizer.vocab_size,
            max_len=cfg.model.max_len,
            embedding_dim=cfg.model.num_hiddens,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            bias=cfg.model.bias,
            eps=cfg.model.eps,
            include_head=False
        )
        if not has_ckpt:
            state = torch.load(cfg.model_path, map_location="cpu")['model']
            self.model.load_state_dict(state, strict=False)

        # Inject rsLoRA + Gated Adapter (only these parameters are trainable)
        apply_hybrid_peft(self.model, rank=cfg.peft_rank)

        # Task output head & loss
        self.prediction_head = CDGPTSequenceTaskHead()
        self.criterion = torch.nn.MSELoss()
        self.save_hyperparameters()

        # Temporary validation output cache
        self._eval_buffer: list[dict[str, torch.Tensor]] = []

    # ---------------------------------------------------------- forward & steps
    def forward(self, batch):
        hids = self.model(batch['input_ids'], attention_mask=batch['attention_mask'])
        return self.prediction_head(batch['input_ids'], hids)

    def _shared_step(self, batch):
        preds = self(batch)
        loss = self.criterion(preds, batch['labels'])
        return loss, preds

    def training_step(self, batch, _):
        loss, _ = self._shared_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, preds = self._shared_step(batch)
        self._eval_buffer.append({'loss': loss, 'preds': preds.detach(),
                                  'labels': batch['labels'].detach()})

    test_step = validation_step  # same logic

    def _flush_eval_buffer(self, stage: str):
        loss = torch.stack([x['loss'] for x in self._eval_buffer]).mean()
        preds = torch.cat([x['preds'] for x in self._eval_buffer]).cpu().numpy()
        labels = torch.cat([x['labels'] for x in self._eval_buffer]).cpu().numpy()
        self._eval_buffer.clear()
        r2 = r2_score(labels, preds)
        rmse = np.sqrt(mean_squared_error(labels, preds))
        self.log(f'{stage}_loss', loss, prog_bar=True)
        self.log(f'{stage}_r2', r2, prog_bar=True)
        self.log(f'{stage}_rmse', rmse, prog_bar=True)

    def on_validation_epoch_end(self):
        self._flush_eval_buffer('val')

    def on_test_epoch_end(self):
        self._flush_eval_buffer('test')

    # -------------------------------------------------------------- optimizers
    def configure_optimizers(self):
        # Different learning rates for LoRA vs Adapter
        param_groups = get_peft_param_groups(self.model,
                                             lr_lora=self.cfg.lr_lora,
                                             lr_adapter=self.cfg.lr_adapter)
        optim = torch.optim.AdamW(param_groups)

        # Linear warm‑up → ReduceLROnPlateau
        sched_warmup = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1,
                                                         total_iters=100)
        sched_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', patience=3, factor=0.5, min_lr=1e-6)
        
        # return {
        #     'optimizer': optim,
        #     'lr_scheduler': [
        #         {'scheduler': sched_warmup, 'interval': 'step'},
        #         {'scheduler': sched_plateau, 'interval': 'epoch', 'monitor': 'val_loss'}
        #     ]
        # }
        return [optim], [
            {
                'scheduler': sched_warmup,
                'interval': 'step',
                'frequency': 1,
                'name': 'warmup'
            },
            {
                'scheduler': sched_plateau,
                'interval': 'epoch',
                'monitor': 'val_loss',
                'frequency': 1,
                'name': 'plateau'
            }
        ]

# -----------------------------------------------------------------------------
# Utility: adaptive batch size
# -----------------------------------------------------------------------------

def get_adaptive_batch_size(model_size='1b', memory_factor=1):
    if not torch.cuda.is_available():
        return 32
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
    mem_gb = gpu.total_memory / 2**30
    # Rough mapping (using 1B as baseline):
    ranges = {(0, 24): 256, (24, 32): 384, (32, 48): 512, (48, 64): 768,
              (64, 96): 1024, (96, float('inf')): 1536}
    factors = {'small': 4, 'base': 2, '1b': 1, '7b': 0.15}
    for (lo, hi), bs in ranges.items():
        if lo <= mem_gb < hi:
            print(f"mem_gb: {mem_gb}, bs: {bs}, factors: {factors.get(model_size, 1)}, memory_factor: {memory_factor}")
            return int(bs * factors.get(model_size, 1) * memory_factor)
    return 32

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Fine‑tune CD‑GPT (Hybrid‑PEFT)')
    p.add_argument('--data_dir', type=str, default='data/')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    p.add_argument('--output_dir', type=str, default='outputs/')
    p.add_argument('--model_size', type=str, default='1b')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=0)
    p.add_argument('--memory_factor', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--peft_rank', type=int, default=16)
    p.add_argument('--lr_lora', type=float, default=1e-4)
    p.add_argument('--lr_adapter', type=float, default=2e-5)
    p.add_argument('--resume_ckpt', type=str, default=None)
    return p.parse_args()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    # cfg baseline
    cfg = get_config()
    cfg.tokenizer.path = os.path.join(args.checkpoint_dir, 'tokenizer.model')
    cfg.model_path = os.path.join(args.checkpoint_dir, f'CD-GPT-{args.model_size}.pth')
    cfg.learning_rate = None  # No longer using unified LR
    cfg.peft_rank = args.peft_rank
    cfg.lr_lora, cfg.lr_adapter = args.lr_lora, args.lr_adapter

    # —— batch size ——
    cfg.batch_size = args.batch_size or get_adaptive_batch_size(
        args.model_size, args.memory_factor)

    # —— datasets ——
    tk = SentencePieceTokenizer(cfg.tokenizer.path)
    paths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)]
    train_ds, val_ds, test_ds = load_and_process_data(paths, 0.7, 0.15, tokenizer=tk)

    collate = train_ds.collate_fn
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate)

    # —— model & trainer ——
    lit_model = CDGPTLightningModule(cfg, has_ckpt=args.resume_ckpt is not None)

    callbacks = [
        pl.callbacks.ModelCheckpoint(dirpath=args.output_dir, monitor='val_r2',
                                     filename='best-{epoch:02d}-{val_r2:.2f}',
                                     save_top_k=1, mode='max'),
        pl.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min'),
        pl.callbacks.LearningRateMonitor('step')
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu', devices=1,
        precision='16-mixed',
        callbacks=callbacks,
        val_check_interval=0.5,
        gradient_clip_val=1.0,
        deterministic=True
    )

    if args.resume_ckpt:
        trainer.fit(lit_model, train_loader, val_loader,
                    ckpt_path=args.resume_ckpt)
    else:
        trainer.fit(lit_model, train_loader, val_loader)
    trainer.test(lit_model, test_loader)


if __name__ == '__main__':
    main()
