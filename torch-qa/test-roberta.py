import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split

# 建立 RoBERTa tokenizer 和模型
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForMaskedLM.from_pretrained("roberta-base")

# 設定訓練資料
data = [
    "This is the first sentence.",
    "Another sentence for training.",
    "A third sentence to be used for training."
]

input_ids = [tokenizer.encode(text, add_special_tokens=True, padding='max_length', max_length=20, truncation=True) for text in data]

# 將 input_ids 轉換為 TensorDataset
input_ids = torch.tensor(input_ids, dtype=torch.long)
dataset = TensorDataset(input_ids, input_ids)  # 使用 input_ids 作為目標序列

# 設定訓練參數
batch_size = 2
num_epochs = 5
learning_rate = 2e-5

# 分割訓練和驗證資料
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 建立 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 建立優化器
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 訓練模型
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, labels = batch
        input_ids = input_ids.to('cuda')  # 如果有 GPU，可以將 Tensor 移到 GPU 上
        labels = labels.to('cuda')
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Training Loss: {avg_train_loss}")

# 儲存模型權重
model.save_pretrained("roberta_trained_model")
tokenizer.save_pretrained("roberta_trained_model")

# 推斷方法
def generate_text(model, input_text):
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(input_text, add_special_tokens=True, padding='max_length', max_length=20, truncation=True)).unsqueeze(0)
        input_ids = input_ids.to('cuda')  # 如果有 GPU，可以將 Tensor 移到 GPU 上
        outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0])
    return generated_text

# 示範推斷
input_text = "This is a test sentence."
generated_text = generate_text(model, input_text)
print("Generated Text:", generated_text)
