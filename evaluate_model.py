#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate CD-GPT model performance and record prediction results
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Import necessary modules
from finetune_CDGPT import CDGPTLightningModule
from model.finetune import CDGPTFineTune, CDGPTSequenceTaskHead
from model.cd_gpt import CDGPT
from tokenizer import SentencePieceTokenizer
from DREAMdataset import load_and_process_data
from config import get_config

# Add fvcore.common.config.CfgNode to safe globals list
from fvcore.common.config import CfgNode
import torch.serialization
torch.serialization.add_safe_globals([CfgNode])

def load_model_and_predict():
    # 1. Load configuration
    cfg = get_config()
    cfg.tokenizer.path = 'checkpoints/tokenizer.model'
    cfg.model_path = 'checkpoints/CD-GPT-1b.pth'
    cfg.peft_rank = 16
    cfg.lr_lora = 1e-4
    cfg.lr_adapter = 2e-5
    
    # Add has_ckpt flag to indicate loading from checkpoint
    cfg.has_ckpt = True

    # 2. Create Lightning model
    model = CDGPTLightningModule(cfg, has_ckpt=True)
    
    # 3. Load model from checkpoint
    ckpt_path = 'outputs/best-epoch=11-val_r2=0.84.ckpt'
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    
    print(f"Model loaded, checkpoint from epoch: {checkpoint['epoch']}")
    
    # 4. Load test dataset
    tokenizer = SentencePieceTokenizer(cfg.tokenizer.path)
    data_dir = 'data/'
    paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    train_ds, val_ds, test_ds = load_and_process_data(paths, 1, 0, tokenizer=tokenizer)
    
    # 5. Create DataLoader
    test_loader = DataLoader(
        train_ds, 
        batch_size=256+128, 
        shuffle=False,
        num_workers=4,
        collate_fn=train_ds.collate_fn
    )
    
    print(f"Test dataset size: {len(train_ds)}")
    
    # 6. Enter evaluation mode
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    # 7. Run predictions
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move data to appropriate device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Run forward pass to get predictions
            predictions = model(batch)
            
            # Collect predictions and labels
            all_preds.append(predictions.cpu())
            all_labels.append(batch['labels'].cpu())
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx}/{len(test_loader)} batches")
    
    # 8. Merge results
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # 9. Calculate evaluation metrics
    r2 = r2_score(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    
    print(f"Evaluation results on test set:")
    print(f"R2 score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # 10. Save prediction results and true labels
    results_df = pd.DataFrame({
        'true_value': all_labels.flatten(),
        'predicted_value': all_preds.flatten()
    })
    
    results_df.to_csv('test_predictions_epoch11_1.csv', index=False)
    print(f"Prediction results saved to 'test_predictions_epoch11_1.csv'")
    
    # 11. Visualize prediction results
    plt.figure(figsize=(10, 6))
    plt.scatter(all_labels, all_preds, alpha=0.3)
    
    # Add perfect prediction line
    min_val = min(all_labels.min(), all_preds.min())
    max_val = max(all_labels.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Model Performance Comparison (Epoch 11)\nR2 = {r2:.4f}, RMSE = {rmse:.4f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_predictions_epoch11_scatter1.png', dpi=300)
    print(f"Scatter plot saved to 'test_predictions_epoch11_scatter1.png'")
    
    return results_df

if __name__ == "__main__":
    results = load_model_and_predict()
    
    # Show first 10 prediction results
    print("\nFirst 10 prediction results:")
    print(results.head(10)) 