from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")

# 假设这是您要嵌入的两篇文章
article1 = "文章1的内容"
article2 = "文章2的内容"

# 将文章编码为模型可以理解的形式
input1 = tokenizer(article1, return_tensors="pt")
input2 = tokenizer(article2, return_tensors="pt")

# 生成嵌入
with torch.no_grad():
    embedding1 = model(input1.input_ids, input1.attention_mask).last_hidden_state
    embedding2 = model(input2.input_ids, input2.attention_mask).last_hidden_state

# 假设这是您的查询文本
query = "查询文本"

# 将查询编码为模型可以理解的形式
input_query = tokenizer(query, return_tensors="pt")

# 生成查询嵌入
with torch.no_grad():
    query_embedding = model(input_query.input_ids, input_query.attention_mask).last_hidden_state

# 计算查询与每篇文章嵌入之间的相似度（例如，使用余弦相似度）
similarity1 = torch.nn.functional.cosine_similarity(query_embedding, embedding1)
similarity2 = torch.nn.functional.cosine_similarity(query_embedding, embedding2)

# 根据相似度选择最相关的文章并生成答案输出
if similarity1 > similarity2:
    answer_input = input1
else:
    answer_input = input2

answer_output = model.generate(answer_input.input_ids, answer_input.attention_mask)
answer_text = tokenizer.decode(answer_output[0], skip_special_tokens=True)

print(answer_text)