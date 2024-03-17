import torch
from torch.utils.data import Dataset
import pandas as pd

# 모델 입력을 위해 데이터를 sequence 단위로 split하는 클래스
class dataloader(Dataset):
    def __init__(self, data, length):
        assert len(data) >= length
        self.data = data
        self.length = length

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx+self.length]).t().to(torch.float32)

    def __len__(self):
        return max(len(self.data)-self.length, 0)
    
