import torch
from torch.utils.data import DataLoader
from DREAMdataset import load_and_process_data
from tokenizer import SentencePieceTokenizer
import os
from config import get_config

def test_dataloader():
    # 配置初始化
    cfg = get_config()
    cfg.tokenizer.path = "checkpoints/tokenizer.model"
    cfg.train_data_path = "data/"
    cfg.batch_size = 512  # 使用较小的batch size便于测试
    cfg.device = "cpu"
    
    # 数据加载
    train_data_paths = os.listdir(cfg.train_data_path)
    train_data_paths = [os.path.join(cfg.train_data_path, path) for path in train_data_paths]
    train_dataset, val_dataset, test_dataset = load_and_process_data(
        train_data_paths, 
        train_ratio=0.7, 
        val_ratio=0.15,
        tokenizer=SentencePieceTokenizer(cfg.tokenizer.path)
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        collate_fn=val_dataset.collate_fn
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        collate_fn=test_dataset.collate_fn
    )
    
    # 测试数据加载
    print("开始测试数据加载...")
    try:
        for batch_idx, batch in enumerate(train_loader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"输入数据形状: {batch['input_ids'].shape}")
            print(f"标签数据形状: {batch['labels'].shape}")
            print(f"输入数据类型: {batch['input_ids'].dtype}")
            print(f"标签数据类型: {batch['labels'].dtype}")
            print(f"输入数据范围: [{batch['input_ids'].min()}, {batch['input_ids'].max()}]")
            print(f"标签数据范围: [{batch['labels'].min():.4f}, {batch['labels'].max():.4f}]")
    except Exception as e:
        breakpoint()

    try:
        for batch_idx, batch in enumerate(val_loader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"输入数据形状: {batch['input_ids'].shape}")
            print(f"标签数据形状: {batch['labels'].shape}")
            print(f"输入数据类型: {batch['input_ids'].dtype}")
            print(f"标签数据类型: {batch['labels'].dtype}")
            print(f"输入数据范围: [{batch['input_ids'].min()}, {batch['input_ids'].max()}]")
            print(f"标签数据范围: [{batch['labels'].min():.4f}, {batch['labels'].max():.4f}]")
    except Exception as e:
        breakpoint()

    try:
        for batch_idx, batch in enumerate(test_loader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"输入数据形状: {batch['input_ids'].shape}")
            print(f"标签数据形状: {batch['labels'].shape}")
            print(f"输入数据类型: {batch['input_ids'].dtype}")
            print(f"标签数据类型: {batch['labels'].dtype}")
            print(f"输入数据范围: [{batch['input_ids'].min()}, {batch['input_ids'].max()}]")
            print(f"标签数据范围: [{batch['labels'].min():.4f}, {batch['labels'].max():.4f}]")
    except Exception as e:
        breakpoint()

    print("\n数据加载测试完成!")

if __name__ == "__main__":
    test_dataloader() 