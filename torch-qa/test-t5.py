import os.path

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

"""
pip install sentencepiece
"""

# 初始化 T5 分詞器和模型
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

pth_file = "./models/t5_translation_model.pt"
if os.path.exists(pth_file):
    model.load_state_dict(torch.load(pth_file))

# 定義訓練資料
train_data = [
    ("translate: 'select id from p",
     "lparam type select; cols id; fromSources p; rparam"),
    ("translate: 'select id,name from p",
     "lparam type select; cols id, name; fromSources p; rparam"),
    ("translate: 'select id,name,birth from p",
     "lparam type select; cols id, name; birth; fromSources p; rparam"),
]

# 訓練設定
batch_size = 1
num_epochs = 10
learning_rate = 3e-4


def train_model():
    global input_text, input_ids, loss
    # 構造訓練迴圈
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for source, target in train_data:
            input_text = f"translate: '{source}'"
            target_text = target

            input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, padding='max_length')
            target_ids = tokenizer.encode(target_text, return_tensors='pt', max_length=512, padding='max_length')

            optimizer.zero_grad()
            loss = model(input_ids=input_ids, labels=target_ids).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")
    # 保存最佳權重
    torch.save(model.state_dict(), pth_file)


train_model()


# 示範推斷方法
def translate(model, source_text):
    model.eval()
    input_text = f"translate: '{source_text}'"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translated_text

# 示範翻譯
input_text = "select addr from cus"
translated_text = translate(model, input_text)
print("Input Text:", input_text)
print("Translated Text:", translated_text)
