import os.path

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import pad_list


class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoding_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, input_size),
            nn.Sigmoid()  # 這裡使用 Sigmoid 函數來保證輸出在 [0, 1] 範圍內
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 定義訓練數據
input_size = 20
encoding_size = 1
data = torch.randn(100, input_size)  # 假設有 100 個 10 長度的序列

# 初始化模型和優化器
autoencoder = Autoencoder(input_size, encoding_size)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 訓練模型
model_pt_file = 'models/autoencoder.pt'
if os.path.exists(model_pt_file):
    print(f"load model pt")
    autoencoder.load_state_dict(torch.load(model_pt_file))
for epoch in range(20000):
    optimizer.zero_grad()
    output = autoencoder(data)
    loss = loss_fn(output, data)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        torch.save(autoencoder.state_dict(), model_pt_file)
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 使用模型進行壓縮和還原
input_seq = pad_list([1, 2, 9, 2, 3, 5, 6, 4, 8, 7], input_size)
input_sequence = torch.tensor(input_seq, dtype=torch.float)
compressed = autoencoder.encoder(input_sequence.unsqueeze(0)).item()
reconstructed = autoencoder.decoder(torch.tensor([[compressed]])).squeeze().detach().numpy()

print("Original Sequence:", input_sequence.numpy())
print("Compressed:", compressed)
print("Reconstructed Sequence:", reconstructed)
