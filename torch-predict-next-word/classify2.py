#您可以輕鬆地將上面的示例修改為多元分類問題。在多元分類問題中，每個樣本可以屬於多個類別。為了解決這種問題，
# 我們可以使用二元交叉熵損失函數（BCEWithLogitsLoss）並將標籤表示為一個獨熱編碼向量。

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
train_labels = [
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0]
]
train_dataset = MyDataset(train_data, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=32)

# 初始化模型並進行訓練
model = GPT1(vocab_size=1000, d_model=512, nhead=8,
             num_layers=6, num_classes=3)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for batch in train_dataloader:
        data, labels = batch
        data = torch.stack(data)
        optimizer.zero_grad()
        output = model(data)
        print(f'{output=}')
        print(f'{labels=}')
        # loss = criterion(output.view(-1), labels.view(-1))
        labels = torch.stack(labels).float()
        #labels = labels.unsqueeze(0)
        #labels = labels.squeeze(0)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

# 定義一個文本分類函數
def classify_text(text):
    text_tensor = torch.tensor(text).unsqueeze(0)
    output = model(text_tensor)
    pred = (output > 0).int().squeeze().tolist()
    return pred

# 測試文本分類函數
text = [1, 2, 3]
pred_classes = classify_text(text)
print(f'Text: {text}, Predicted classes: {pred_classes}')