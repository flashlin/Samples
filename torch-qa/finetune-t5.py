import os.path

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
# 一般來說下載的模型是
# c:\users\flash\.cache\huggingface\hub

# 定义微调数据集，示例包含三个问题和答案
train_data = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "Who wrote the play 'Romeo and Juliet'?", "answer": "William Shakespeare"},
    {"question": "How to add a new B2B2C domain?", "answer": """Assuming that NOC is ready to provide the following domain for us to add: abc.net
Here are the steps to add a new b2b2c domain:
1. Go to the http://dba-sb-prod.coreop.net/ Artimes website.
   Check whether these domains already exist in the InfoDb.AuthroizedDomain table.
2. Go to the https://forseti-api-a.sbotopex.com/swagger/index.html website.
3. If "abc.net" already exists in the InfoDb.AuthroizedDomain table:
   Perform a POST request to /api/public/AdminAuthorizedDomain/DeleteDomain to delete "abc.net" from the database.
4. Perform a POST request to /api/public/AdminAuthorizedDomain/InsertDomain to insert "abc.net" domain.
5. After the insertion is successful, check the result at http://forseti-api-a.sbotopex.com/domain.
6. If "abc.net" is intended for use in China, set the domain to "disable" in the database to prevent it from being used by agents from other countries.
7. If "abc.net" uses the HTTPS protocol, go to the http://dragonballz.coreop.net/ website and modify the AcceptHttpsDomain key value in the global_settings of the Forseti website to include "abc.net".
8. In the git Forseti pipeline, select "production:reload" to reload the global_settings configuration.
9. Restart the pods of the following projects:
   - Tera Backend
   - Pollux (SG Support or Hinoki should execute the necessary commands to restart pollux)
"""}
]

# 加载Flan-T5预训练模型和分词器
model_name = "google/flan-t5-large"
model_name = "google/flan-t5-xl"   
model_name = "google/flan-t5-xxl"   # seq_len=4096 似乎需要 80GB VRAM
model_name = "google/flan-t5-base"  # seq_len=512
model_name = "google/flan-t5-small"
#--------------------------------------------------
model_name = "google/flan-t5-xl"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
print("tokenizer loaded")
device = 'cpu'
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map=device) #, torch_dtype=torch.float16)
print("model loaded")


def get_data_loader(data):
    # 将问题和答案转换为模型输入格式
    inputs = tokenizer([item["question"] for item in data], padding=True, truncation=True, return_tensors="pt")
    labels = tokenizer([item["answer"] for item in data], padding=True, truncation=True, return_tensors="pt")

    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(inputs.input_ids, inputs.attention_mask, labels.input_ids, labels.attention_mask)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    return data_loader


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
pt_file = 'models/flan-t5.pt'

def train(model, data_loader, loss_fn, optimizer):
    model.train()
    for batch in data_loader:
        input_ids, attention_mask, label_ids, label_attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label_ids = label_ids.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)
        loss = outputs.loss
        print(f"{loss=}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pth_file = pt_file
    torch.save(model.state_dict(), pth_file)


# 进行微调训练
if os.path.exists(pt_file):
    model.load_state_dict(torch.load(pt_file))
    print(f"load {pt_file}")
data_loader = get_data_loader(train_data)

for epoch in range(10):
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
question = "how to add new b2b2c domain?"
answer = generate_answer(model, question)
print("Question:", question)
print("Answer:", answer)