

train_data = [
    ("I love this product!", 1),   # 正面情感，標籤為 1
    ("This movie is terrible.", 0),  # 負面情感，標籤為 0
    ("The weather is beautiful.", 1)  # 正面情感，標籤為 1
]

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 初始化 BERT 分詞器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# 訓練設定
batch_size = 1
num_epochs = 5
learning_rate = 2e-5

# 構造訓練迴圈
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for sentence, label in train_data:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        labels = torch.tensor([label], dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_data)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

# 保存最佳權重
#torch.save(model.state_dict(), "bert_sentiment_classification_model.pt")


def predict_sentiment(model, sentence):
    model.eval()
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    return predicted_label

# 示範推斷
input_sentence = "This product is amazing!"
predicted_label = predict_sentiment(model, input_sentence)
print("Input Sentence:", input_sentence)
print("Predicted Label (0=negative, 1=positive):", predicted_label)
