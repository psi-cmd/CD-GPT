import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import product


def build_mapping():
    TF_Ids = ['Cebpb', 'Egr2', 'Esr1', 'Foxj2', 'Foxo1', 'Foxo3', 'Foxo4',
       'Foxp1', 'Foxp2', 'Gmeb2', 'Irf2', 'Junb', 'Mecp2', 'Nr2c1',
       'Pou3f1', 'Sox14', 'Sp1', 'Tbx3', 'Tcf3', 'Zscan20']
    
    more_TF_Ids = ['TF_1', 'TF_2', 'TF_3', 'TF_4', 'TF_5', 'TF_6', 'TF_7', 'TF_8',
       'TF_9', 'TF_10', 'TF_11', 'TF_12', 'TF_13', 'TF_14', 'TF_15',
       'TF_16', 'TF_17', 'TF_18', 'TF_19', 'TF_20', 'TF_21', 'TF_22',
       'TF_23', 'TF_24', 'TF_25', 'TF_26', 'TF_27', 'TF_28', 'TF_29',
       'TF_30', 'TF_31', 'TF_32', 'TF_33', 'TF_34', 'TF_35', 'TF_36',
       'TF_37', 'TF_38', 'TF_39', 'TF_40', 'TF_41', 'TF_42', 'TF_43',
       'TF_44', 'TF_45', 'TF_46', 'TF_47', 'TF_48', 'TF_49', 'TF_50',
       'TF_51', 'TF_52', 'TF_53', 'TF_54', 'TF_55', 'TF_56', 'TF_57',
       'TF_58', 'TF_59', 'TF_60', 'TF_61', 'TF_62', 'TF_63', 'TF_64',
       'TF_65', 'TF_66']
    
    TF_Ids = TF_Ids + more_TF_Ids
    
    ArrayTypes = ["HK", "ME"]
    single_chars = [
        'V', 'R', 'MYR', 'MYL', 'MYG', 'MY', 'MWR', 'MWL', 'MWG', 'MW', 'MVV', 
        'MVR', 'MVQ', 'MVM', 'MVL', 'MVK', 'MVI', 'MVH', 'MVG', 'MVF', 'MVD', 
        'MVC', 'MV', 'MTV', 'MTT', 'MTSS', 'MTSP', 'MTSL', 'MTR', 'MTQ', 'MTP',
        'MTN', 'MTM', 'MTL', 'MTK', 'MTI', 'MTH', 'MTG', 'MTF', 'MTE', 'MTD',
        'MTC', 'MTA', 'MSY', 'MSW', 'MSV', 'MSTV', 'MSTL', 'MST', 'MSSL', 'MSSG',
        'MSS', 'MSRL', 'MSR', 'MSQ', 'MSP', 'MSN', 'MSM', 'MSLL', 'MSL', 'MSKL',
        'MSK', 'MSI', 'MSH', 'MSGL', 'MSGG', 'MSG', 'MSF', 'MSEL', 'MSEK', 'MSEE',
        'MSE', 'MSD', 'MSC', 'MSAL', 'MSA', 'MRV', 'MRR', 'MRQ', 'MRN', 'MRM',
        'MRLL', 'MRL', 'MRK', 'MRI', 'MRH', 'MRG', 'MRF', 'MRD', 'MRC', 'MR',
        'MQV', 'MQR', 'MQQ', 'MQL', 'MQK', 'MQG', 'MQF', 'MQD', 'MQ', 'MPY',
        'MPV', 'MPT', 'MPR', 'MPQ', 'MPP', 'MPN', 'MPM', 'MPL', 'MPK', 'MPI',
        'MPH', 'MPG', 'MPF', 'MPE', 'MPD', 'MPA', 'MNY', 'MNV', 'MNT', 'MNR',
        'MNQ', 'MNP', 'MNN', 'MNM', 'MNL', 'MNK', 'MNI', 'MNH', 'MNG', 'MNF',
        'MND', 'MNC', 'MN', 'MMV', 'MMR', 'MMQ', 'MML', 'MMK', 'MMG', 'MMD',
        'MM', 'MLV', 'MLR', 'MLLL', 'MLL', 'MLK', 'MLG', 'MLC', 'ML', 'MKY',
        'MKV', 'MKR', 'MKQ', 'MKN', 'MKM', 'MKLL', 'MKL', 'MKK', 'MKI', 'MKH',
        'MKG', 'MKF', 'MKD', 'MKC', 'MK', 'MIV', 'MIR', 'MIQ', 'MIN', 'MIL', 'MIK'
    ]
    mapping = {k: v for k, v in zip(product(TF_Ids, ArrayTypes), single_chars)}
    return mapping

_mapping = build_mapping()

def TF_ArrayType_to_small_token(TF_Id, ArrayType):
    """should minimize the token length, so map combination of TF_Id and ArrayType to a small token"""
    if (TF_Id, ArrayType) in _mapping:
        return _mapping[(TF_Id, ArrayType)]
    else:
        import pickle
        pickle.dump(_mapping, open("mapping.pkl", "wb"))
        pickle.dump(TF_Id, open("TF_Id.pkl", "wb"))
        pickle.dump(ArrayType, open("ArrayType.pkl", "wb"))
        raise ValueError(f"TF_Id: {TF_Id}, ArrayType: {ArrayType} is not in the allowed list")

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
        self.device = "cpu"  # lightning will move the data to the correct device

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
            text = f"{TF_ArrayType_to_small_token(TF_Id, array_type)}{sequence}"
            
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
            sample_mask = attention_mask_2d[i].bool()
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))
            combined_mask = causal_mask & sample_mask.unsqueeze(0) & sample_mask.unsqueeze(1)
            attention_mask_4d[i, 0, :seq_len, :seq_len] = combined_mask

        return {
            'input_ids': torch.stack(batch_encodings),
            'attention_mask': attention_mask_4d,
            'labels': torch.tensor(batch_labels)
        }