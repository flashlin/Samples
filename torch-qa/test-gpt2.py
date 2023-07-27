import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import pad_list
from sql_network import sql_to_value, SqlTrainDataset, pad_collate_fn, key_dict

"""
GPT-2 是 OpenAI 提出的一種基於 Transformer 架構的語言模型，它共有4個版本，分別是：
GPT-2 "Small" (hidden_size=768, num_layers=12)
GPT-2 "Medium" (hidden_size=1024, num_layers=24)
GPT-2 "Large" (hidden_size=1280, num_layers=36)
GPT-2 "XL" (hidden_size=1600, num_layers=48)
"""
class TinyGPT2(nn.Module):
    def __init__(self, vocab_size, hidden_size=64, num_layers=2):
        super(TinyGPT2, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=2) for _ in range(num_layers)
        ])

        # Final linear layer for output
        self.out_linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output = embedded

        for layer in self.transformer_layers:
            output = layer(output)

        output = self.out_linear(output)
        return output


import torch.optim as optim


def train_model(model, train_loader, num_epochs=100, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "./models/gpt2.pt")

    print("Training complete!")


def generate_text(model, start_text, max_length=100):
    model.eval()
    input_ids = sql_to_value(start_text)
    # input_ids = pad_list(input_ids, max_length)
    input_ids = [key_dict['<s>']] + input_ids + [key_dict['</s>']]
    input_ids = torch.as_tensor(input_ids, dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_length):
            output_ids = model(input_ids)
            next_token_id = output_ids[:, -1].argmax().unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    output_ids = input_ids
    return output_ids


from torch.utils.data import DataLoader, TensorDataset

# 假設我們的資料是以 ID 序列表示的
data = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
]

new_data = []
for item in data:
    item = pad_list(item, 100)
    new_data.append(item)
data = new_data

# 訓練資料轉換成 DataLoader
# train_data = TensorDataset(torch.LongTensor(data), torch.LongTensor(data))
# train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

dataset = SqlTrainDataset("./train_data/sql.txt", max_seq_len=100)
train_loader = DataLoader(dataset, batch_size=2, collate_fn=pad_collate_fn)


# 建立模型和 tokenizer
vocab_size = 10000  # 假設有16個詞彙
model = TinyGPT2(vocab_size)

pth_file = './models/gpt2.pth'
if os.path.exists(pth_file):
    model.load_state_dict(torch.load(pth_file))


# 訓練模型
train_model(model, train_loader)

# 進行推斷
start_text = "select id from p"
generated_text = generate_text(model, start_text)
print("Generated Text:", generated_text)

