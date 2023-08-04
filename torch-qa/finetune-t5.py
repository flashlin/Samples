import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 定义微调数据集，示例包含三个问题和答案
train_data = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "Who wrote the play 'Romeo and Juliet'?", "answer": "William Shakespeare"}
]

# 加载Flan-T5预训练模型和分词器
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
print("tokenizer loaded")

model = T5ForConditionalGeneration.from_pretrained(model_name)
print("model loaded")
# 定义微调数据加载器
def get_data_loader(data):
    # 将问题和答案转换为模型输入格式
    inputs = tokenizer([item["question"] for item in data], padding=True, truncation=True, return_tensors="pt")
    labels = tokenizer([item["answer"] for item in data], padding=True, truncation=True, return_tensors="pt")

    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(inputs.input_ids, inputs.attention_mask, labels.input_ids, labels.attention_mask)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    return data_loader

# 定义损失函数
loss_fn = torch.nn.CrossEntropyLoss()

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# 进行微调
def train(model, data_loader, loss_fn, optimizer):
    model.train()
    for batch in data_loader:
        input_ids, attention_mask, label_ids, label_attention_mask = batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pth_file = 'models/flant-t5-small.pth'
    torch.save(model.state_dict(), pth_file)

# 进行微调训练
data_loader = get_data_loader(train_data)
train(model, data_loader, loss_fn, optimizer)

# 进行推理
def generate_answer(model, question):
    model.eval()
    inputs = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=512)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# 给定一个问题，生成相应的答案
question = "What is the capital of Italy?"
answer = generate_answer(model, question)
print("Question:", question)
print("Answer:", answer)
