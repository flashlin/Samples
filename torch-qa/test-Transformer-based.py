import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_utils import pad_list
from sql_network import SqlTrainDataset, pad_collate_fn, sql_to_value, key_dict


# 定義 Transformer-based 翻譯模型
class Translator(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model=256, num_heads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(Translator, self).__init__()
        self.source_embedding = nn.Embedding(source_vocab_size, d_model)
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, num_heads, num_encoder_layers, num_decoder_layers)
        self.fc = nn.Linear(d_model, target_vocab_size)

    def forward(self, source, target):
        source = self.source_embedding(source)
        target = self.target_embedding(target)
        output = self.transformer(source, target)
        output = self.fc(output)
        return output

# 初始化模型並設置優化器
source_vocab_size = 10000
target_vocab_size = 10000
model = Translator(source_vocab_size, target_vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)


dataset = SqlTrainDataset("./train_data/sql.txt", max_seq_len=100)
train_loader = DataLoader(dataset, batch_size=2, collate_fn=pad_collate_fn)



# 訓練模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()

    for inputs, target_ids in train_loader:
        optimizer.zero_grad()
        output = model(inputs, target_ids)

        # print(f"{output.shape=}")
        # print(f"{target_ids.shape=}")
        # output_reshape = output.reshape(-1, target_vocab_size)
        # print(f"{output_reshape.shape=}")
        # target_reshape = target_ids[:, 1:].reshape(-1)
        # print(f"{target_reshape.shape=}")

        loss = nn.CrossEntropyLoss(ignore_index=0)(output.reshape(-1, target_vocab_size), target_ids[:, :].reshape(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item()}")

# 示範推斷方法
def translate(model, source_text):
    model.eval()
    source_ids = sql_to_value(source_text)
    source_ids = pad_list(source_ids, 100)
    source_ids = torch.tensor(source_ids, dtype=torch.long).unsqueeze(0)
    target_ids = pad_list([key_dict['<s>']], 100)
    target_ids = torch.tensor(target_ids, dtype=torch.long).unsqueeze(0)  # 開始符號 <sos>

    end_index = key_dict['</s>']
    with torch.no_grad():
        for _ in range(20):  # 限制生成的句子長度為 20 個詞
            print(f"{source_ids.shape=}")
            # TODO: 不知道為什麼
            output = model(source_ids, None)
            next_word_id = output.argmax(dim=-1)[:, -1].item()
            target_ids = torch.cat([target_ids, torch.tensor([[next_word_id]], dtype=torch.long)], dim=-1)
            if next_word_id == end_index:  # 結束符號 <eos>
                break
    translated_text = target_ids[0]
    return translated_text

# 示範翻譯
input_text = "I love pytorch."
translated_text = translate(model, input_text)
print("Input Text:", input_text)
print("Translated Text:", translated_text)
