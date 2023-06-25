import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from id_utils import generate_random_id_for_train


def tensor2d(inputs: [[int]], data_type=np.float32) -> torch.Tensor:
    np_array = np.array(inputs, dtype=data_type)
    input_tensor = torch.from_numpy(np_array)
    return input_tensor


# 定義數據集
class MyDataset(Dataset):
    def __init__(self):
        self.samples = []
        for n in range(100):
            id_str, train_id9, check_code = generate_random_id_for_train()
            self.samples.append((train_id9, check_code))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        train_id9, check_code = self.samples[idx]
        return tensor2d(train_id9), torch.tensor(check_code)


dataset = MyDataset()
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)



