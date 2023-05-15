import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class GPT1(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        super(GPT1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


# 定義訓練數據集
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 假設我們有以下訓練數據和標籤
train_data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
train_labels = [0, 1, 2]
train_dataset = MyDataset(train_data, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=32)

# 初始化模型並進行訓練
model = GPT1(vocab_size=1000, d_model=512, nhead=8,
             num_layers=6, num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for batch in train_dataloader:
        data, labels = batch
        data = torch.stack(data)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, 5), labels.view(-1))
        loss.backward()
        optimizer.step()


# 定義一個文本分類函數
def classify_text(text):
    text_tensor = torch.tensor(text).unsqueeze(0)
    output = model(text_tensor)
    pred = output.argmax(dim=1).item()
    return pred


# 測試文本分類函數
text = [1, 2, 3]
pred_class = classify_text(text)
print(f'Text: {text}, Predicted class: {pred_class}')
