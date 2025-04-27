from model.finetune import CDGPTFineTune
from config import get_config
import torch
from torch.utils.data import DataLoader
from DREAMdataset import load_and_process_data
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
import argparse

from tokenizer import SentencePieceTokenizer
from model.finetune import CDGPTSequenceTaskHead

class CDGPTLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        state = torch.load(cfg.model_path, map_location="cuda")["model"]
        self.model = CDGPTFineTune(cfg, use_lora=True, use_adapter=True)
        self.model.load_state_dict(state, strict=False)
        
        # 损失函数
        self.criterion = torch.nn.MSELoss()
        self.save_hyperparameters()

        self.prediction_head = CDGPTSequenceTaskHead()

        self.validation_test_step_outputs = []

    def forward(self, x):
        hiddens = self.model(x['input_ids'], attention_mask=x['attention_mask'])
        output = self.prediction_head(x['input_ids'], hiddens)
        return output
    

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch['labels'])
        
        # 记录训练指标
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch['labels'])
        
        self.validation_test_step_outputs.append({
            'val_loss': loss,
            'preds': outputs.detach().cpu(),
            'labels': batch['labels'].detach().cpu()
        })

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_test_step_outputs]).mean()
        all_preds = np.concatenate([x['preds'].numpy() for x in self.validation_test_step_outputs])
        all_labels = np.concatenate([x['labels'].numpy() for x in self.validation_test_step_outputs])
        
        # 计算评估指标
        r2 = r2_score(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        
        # 记录验证指标
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_r2', r2, prog_bar=True)
        self.log('val_rmse', rmse, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_test_step_outputs]).mean()
        all_preds = np.concatenate([x['preds'].numpy() for x in self.validation_test_step_outputs])
        all_labels = np.concatenate([x['labels'].numpy() for x in self.validation_test_step_outputs])
        
        r2 = r2_score(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        
        self.log('test_loss', avg_loss)
        self.log('test_r2', r2)
        self.log('test_rmse', rmse)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
    
        # 创建两个scheduler
        warmup_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=100
            ),
            "interval": "step",
            "frequency": 1,
            "name": "warmup"
        }
        
        plateau_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=3,
                factor=0.5,
                min_lr=1e-6
            ),
            "interval": "epoch",
            "monitor": "val_loss",
            "frequency": 1,
            "name": "plateau"
        }
        
        # 直接返回优化器和调度器列表
        return [optimizer], [warmup_scheduler, plateau_scheduler]

def parse_args():
    parser = argparse.ArgumentParser(description='CD-GPT Fine-tuning')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=320, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--model_size', type=str, default='1b', help='模型大小')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--max_length', type=int, default=100, help='最大序列长度')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    
    # 路径相关参数
    parser.add_argument('--data_dir', type=str, default='data/', help='训练数据路径')
    parser.add_argument('--output_dir', type=str, default='checkpoints/', help='输出目录')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='检查点目录')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    pl.seed_everything(args.seed)
    
    # 配置和初始化
    cfg = get_config()
    cfg.tokenizer.path = os.path.join(args.checkpoint_dir, "tokenizer.model")
    cfg.model_path = os.path.join(args.checkpoint_dir, f"CD-GPT-{args.model_size}.pth")
    cfg.train_data_path = args.data_dir
    cfg.batch_size = args.batch_size
    cfg.num_epochs = args.epochs
    cfg.learning_rate = args.learning_rate
    cfg.max_length = args.max_length
    cfg.num_workers = args.num_workers

    # 数据加载
    train_data_paths = os.listdir(cfg.train_data_path)
    train_data_paths = [os.path.join(cfg.train_data_path, path) for path in train_data_paths]
    train_dataset, val_dataset, test_dataset = load_and_process_data(
        train_data_paths, 
        train_ratio=0.7, 
        val_ratio=0.15,
        tokenizer=SentencePieceTokenizer(cfg.tokenizer.path)
    )
    
    # 使用配置中的batch_size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=val_dataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=test_dataset.collate_fn
    )
    
    # 定义模型
    model = CDGPTLightningModule(cfg)

    # def print_dtype(module, input, output):
    #     print(module, "input dtype:", input[0].dtype, "output dtype:", output.dtype)

    # for module in model.modules():
    #     module.register_forward_hook(print_dtype)
    
    # 定义回调函数
    callbacks = [
        ModelCheckpoint(
            monitor='val_r2',
            dirpath=args.output_dir,
            filename='best-model-{epoch:02d}-{val_r2:.2f}',
            save_top_k=1,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min'
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step')  # 添加学习率监控
    ]
    
    # 训练器配置
    trainer = pl.Trainer(
        max_epochs=cfg.num_epochs,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',  # 使用FP16混合精度
        callbacks=callbacks,
        val_check_interval=0.5,
        gradient_clip_val=1.0,  # 添加梯度裁剪
        deterministic=True  # 确保结果可复现
    )
    
    # 训练和测试
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main() 