import torch
from torch.utils.data import Dataset
from transformers import DistilBertModel, DistilBertTokenizer

# 建立 DistilBERT 模型
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
# 建立 DistilBERT 詞彙表
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# 訓練方法
def train(model, tokenizer, train_data, epochs=10):
  """訓練 DistilBERT 模型。

  Args:
    model: DistilBERT 模型。
    tokenizer: DistilBERT 詞彙表。
    train_data: 訓練資料。
    epochs: 訓練週期。

  Returns:
    訓練好的 DistilBERT 模型。
  """
  # 建立數據集迭代器。
  train_loader = torch.utils.data.DataLoader(
      train_data, batch_size=32, shuffle=True
  )

  # 建立損失函數和優化器。
  loss_function = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters())

  # 開始訓練。
  for epoch in range(epochs):
    for batch in train_loader:
      # 將輸入和目標轉換為張量。
      input_ids = batch["input_ids"].to(torch.long)
      attention_mask = batch["attention_mask"].to(torch.long)
      labels = batch["labels"].to(torch.long)

      # 前向傳遞。
      outputs = model(input_ids, attention_mask)

      # 計算損失。
      loss = loss_function(outputs, labels)

      # 反向傳播。
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # 在每個週期結束時評估模型。
    val_loss, val_acc = evaluate(model, tokenizer, val_data)
    print(f"Epoch {epoch + 1}: val_loss = {val_loss}, val_acc = {val_acc}")

  return model

# 推斷方法
def predict(model, tokenizer, input_text):
  """使用 DistilBERT 模型進行推斷。

  Args:
    model: DistilBERT 模型。
    tokenizer: DistilBERT 詞彙表。
    input_text: 輸入文字。

  Returns:
    模型的輸出。
  """

  # 將輸入文字轉換為詞彙表編號。
  input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(torch.long)
  attention_mask = tokenizer(input_text, return_tensors="pt").attention_mask.to(torch.long)

  # 前向傳遞。
  outputs = model(input_ids, attention_mask)

  # 解碼輸出。
  predictions = torch.argmax(outputs, dim=-1)

  # 將詞彙表編號轉換回文字。
  predictions = tokenizer.decode(predictions)

  return predictions

# 3 筆訓練資料示範
train_data = [
  (
    "今天天氣很晴朗",
    "天氣晴朗",
  ),
  (
    "我很高興",
    "我高興",
  ),
  (
    "我要去上學了",
    "我要去上學",
  ),
]

class SqlTrain2Dataset(Dataset):
    def __init__(self):
        data = []
        for src, tgt in train_data:
            src_ids = tokenizer(src, return_tensors="pt").input_ids.to(torch.long)
            tgt_ids = tokenizer(tgt, return_tensors="pt").input_ids.to(torch.long)
            data.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        src, tgt = self.data[index]
        return src, tgt


# 訓練模型
model = train(model, tokenizer, SqlTrain2Dataset())

# 推斷輸出
output = predict(model, tokenizer, "今天天氣很晴朗")
print(output)
