import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_utils import pad_list

"""
B，處於一個詞語的開始
M，處於一個詞語的中間
E，處於一個詞語的末尾
S，單個字
O，未知
原文網址：https://kknews.cc/code/abne4a6.html
"""


def is_integer_or_float(word: str):
    pattern = r'^[-+]?\d+(\.\d+)?$'
    return re.match(pattern, word) is not None


def label_word(word: str) -> str:
    result = []
    if len(word) == 1 or is_integer_or_float(word):
        return 'S'
    for idx, ch in enumerate(word):
        if idx == 0:
            result.append('B')
            continue
        if idx == len(word) - 1:
            result.append('E')
            continue
        result.append('M')
    return ''.join(result)


def text_to_words(text: str) -> list[str]:
    return text.split(' ')


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=128, tagset_size=5):
        """
        :param vocab_size:
        :param embedding_dim: 詞向量維度
        :param hidden_dim:
        :param tagset_size: 標記集大小
        """
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeds(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space


word_to_idx = {"<PAD>": 0, "<UNK>": 1}  # 加入PAD和UNK用于填充和未知词
label_to_idx = {"B": 0, "M": 1, "E": 2, "S": 3, "O": 4}


def read_train_text_file(file: str):
    train_data = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            words = text_to_words(line)
            labels = [label_word(word) for word in words]
            train_data.append((''.join(words), ''.join(labels)))
    return train_data


train_data = read_train_text_file("./train_data/train_small.txt")

for sentence, labels in train_data:
    for word in sentence:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def my_collate(batch):
    inputs = []
    labels = []
    max_seq_len = 0
    for sentence, label in batch:
        max_seq_len = max(len(sentence), max_seq_len)
        max_seq_len = max(len(label), max_seq_len)
    for sentence, label in batch:
        input_value = [word_to_idx[word] for word in sentence]
        label_value = [label_to_idx[ch] for ch in label]
        input_value = pad_list(input_value, max_seq_len)
        label_value = pad_list(label_value, max_seq_len)
        inputs.append(input_value)
        labels.append(label_value)
    return inputs, labels


batch_size = 2
train_loader = DataLoader(MyDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=my_collate)

vocab_size = len(word_to_idx)
tagset_size = 5
model = BiLSTM_CRF(tagset_size=tagset_size)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
criterion = nn.CrossEntropyLoss()

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels)
        # 前向傳遞
        logits = model(inputs)
        # 計算損失
        loss = criterion(logits.view(-1, tagset_size), labels.view(-1))
        # 反向傳遞和參數更新
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
#
# torch.save(model.state_dict(), saved_model_dir)
