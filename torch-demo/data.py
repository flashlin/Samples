from torch.utils.data import Dataset, DataLoader
import numpy as np

# 定義數據集
class MyDataset(Dataset):
    def __init__(self):
        self.samples = [
            (np.random.rand(9), np.random.randint(0, 10)),
            (np.random.rand(9), np.random.randint(0, 10)),
            (np.random.rand(9), np.random.randint(0, 10))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# 創建數據集和數據加載器
dataset = MyDataset()
trainloader = DataLoader(dataset, batch_size=1, shuffle=True)
