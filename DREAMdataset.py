import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_process_data(data_paths: list[str], train_ratio=0.7, val_ratio=0.15, tokenizer=None):
    """
    Load data from multiple files and process it into train/val/test sets.
    Args:
        data_paths (list[str]): List of paths to data files
        train_ratio (float): Ratio for training set (default: 0.7)
        val_ratio (float): Ratio for validation set (default: 0.15)
        tokenizer: Tokenizer for converting text to tensors
        
    Returns:
        tuple: Train, validation and test DataFrames
    """
    # 加载并合并所有数据文件
    dfs = [pd.read_csv(path, sep='\t') for path in data_paths]
    df = pd.concat(dfs, ignore_index=True)
    print(f"{df.shape} data in total.")
    
    # 计算探针偏差（所有TF的平均信号强度）
    # 创建一个探针ID（可以是序列本身作为唯一标识符）
    df['probe_id'] = df['Sequence']
    
    # 计算每个探针在所有TF中的平均信号强度（预期信号）
    probe_bias = df.groupby('probe_id')['Signal_Mean'].mean().reset_index()
    probe_bias.columns = ['probe_id', 'expected_signal']
    
    # 将偏差信息合并回主数据框
    df = pd.merge(df, probe_bias, on='probe_id', how='left')
    
    # 计算校正后的信号值：原始信号除以预期信号，并进行log2变换
    # 避免零除和log(0)问题
    epsilon = 1e-10  # 小的正数防止除零
    df['corrected_signal'] = np.log2((df['Signal_Mean'] + epsilon) / (df['expected_signal'] + epsilon))
    
    # 使用校正后的信号作为标签
    df['normalized_signal'] = df['corrected_signal']
    
    # 根据TF_Id和ArrayType分组进行分层采样
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for (tf_id, array_type), group in df.groupby(['TF_Id', 'ArrayType']):
        # 首先分出训练集
        try:
            train_group, temp_group = train_test_split(
                group, 
                train_size=train_ratio,
                stratify=pd.qcut(group['normalized_signal'], q=4, duplicates='drop').cat.codes
            )
        except ValueError as e:
            print(f"TF_Id: {tf_id}, ArrayType: {array_type} has less than 10 samples, skipped")
            continue
        
        # 从剩余数据中分出验证集和测试集
        val_size = val_ratio / (1 - train_ratio)  # 调整验证集比例
        val_group, test_group = train_test_split(
            temp_group,
            train_size=val_size,
            stratify=pd.qcut(temp_group['normalized_signal'], q=4, duplicates='drop').cat.codes
        )
        
        train_dfs.append(train_group)
        val_dfs.append(val_group)
        test_dfs.append(test_group)
        
    # 创建数据集对象
    train_dataset = DREAMDataset(pd.concat(train_dfs), tokenizer=tokenizer)
    val_dataset = DREAMDataset(pd.concat(val_dfs), tokenizer=tokenizer) 
    test_dataset = DREAMDataset(pd.concat(test_dfs), tokenizer=tokenizer)
    
    return train_dataset, val_dataset, test_dataset

class DREAMDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_length=256, device="cuda"):
        """
        Initialize the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the data
            tokenizer: Tokenizer for converting text to tensors
            max_length (int): Maximum sequence length
            device (str): Device to run the model on
        """
        self._data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return len(self._data)

    def __getitem__(self, idx):
        """
        Get a data sample for the given index.

        Args:
            idx (int): Index of the data sample.

        Returns:
            tuple: Tuple containing input tensor and label.
        """
        try:
            assert isinstance(idx, int), f"idx 必须是 int 类型，但收到 {type(idx)}"
            row = self._data.iloc[idx]
            TF_Id = row['TF_Id']
            array_type = row['ArrayType']
            sequence = row['Sequence'][:35]    
            text = f"<{TF_Id}><{array_type}>{sequence}"
            
            # 使用校正后的信号值作为标签
            label = row['normalized_signal'].astype(np.float32)
        except Exception as e:
            print(f"Type of idx: {type(idx)}")
            print(f"Error at index {idx}: {e}")
            raise e

        encoding = self.tokenizer.encode(text, eos=False, device=self.device)
        mask = torch.ones_like(encoding)
        return encoding, mask, label
    
    def collate_fn(self, batch):
        # 找到当前batch中最长的序列长度
        max_len = max(len(seq) for seq, _, _ in batch)
        
        batch_encodings = []
        batch_masks = []
        batch_labels = []
        
        for seq, mask, label in batch:
            cur_len = len(seq)
            if cur_len < max_len:
                pad_length = max_len - cur_len
                # 对encoding进行padding
                padding = torch.full((pad_length,), self.tokenizer.pad, dtype=seq.dtype, device=seq.device)
                seq = torch.cat([seq, padding])
                # 对mask进行padding，padding部分为0，表示不关注这些位置
                mask_padding = torch.zeros(pad_length, dtype=mask.dtype, device=mask.device)
                mask = torch.cat([mask, mask_padding])
            
            batch_encodings.append(seq)
            batch_masks.append(mask)
            batch_labels.append(label)
        
        attention_mask_2d = torch.stack(batch_masks)
        batch_size, seq_len = attention_mask_2d.shape

        attention_mask_4d = torch.zeros(batch_size, 1, seq_len, seq_len, device=self.device)

        for i in range(batch_size):
            sample_mask = attention_mask_2d[i]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))

            valid_tokens = sample_mask.bool()
            causal_mask = causal_mask * valid_tokens.unsqueeze(0) * valid_tokens.unsqueeze(1)

            attention_mask_4d[i, 0, :seq_len, :seq_len] = causal_mask

        return {
            'input_ids': torch.stack(batch_encodings),
            'attention_mask': attention_mask_4d,
            'labels': torch.tensor(batch_labels)
        }