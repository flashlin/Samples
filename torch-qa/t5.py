from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

torch.hub.set_dir('models')

model_name = 't5-3b'  # 't5-11b' #'t5-base'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=1024)

question = "How to create vue3 app?"
filepath = 'data/vue3.txt'
with open(filepath, 'r', encoding='utf-8') as f:
    large_content = f.read()

# 將大的內容分段
max_chunk_size = 512  # 每個分段的最大長度
chunks = [large_content[i:i+max_chunk_size] for i in range(0, len(large_content), max_chunk_size)]

answers = []
for chunk in chunks:
    input_text = f"question: {question} context: {chunk}"

    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_new_tokens=50)
    answer = tokenizer.decode(outputs[0])
    answers.append(answer)

# 將答案合併
combined_answer = " ".join(answers)
print("Combined Answer:", combined_answer)
