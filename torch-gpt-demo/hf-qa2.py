from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-chinese")

# 將文章轉換為嵌入索引
def embed_text(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        last_hidden_states = model.bert(input_ids)[0]
    return last_hidden_states

# 執行問答
def answer_question(question, context):
    input_text = "[CLS] " + question + " [SEP] " + context + " [SEP]"
    input_ids = tokenizer.encode(input_text)
    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
    return answer

# 選擇最相似的文章作為上下文
def select_context(question, embeddings):
    question_embedding = embed_text(question)
    similarities = [cosine_similarity(embedding, question_embedding) for embedding in embeddings]
    most_similar_index = similarities.index(max(similarities))
    return most_similar_index

# 示例文章
text1 = "MingMing's birth is 2023-03-01"
text2 = "小明有哥哥, 哥哥的名字是Flash"
text3 = "企鵝住在南極"

# 將文章轉換為嵌入索引
embedding1 = embed_text(text1)
embedding2 = embed_text(text2)
embedding3 = embed_text(text3)

# 選擇最相似的文章作為上下文
question = "When is MingMing born?"
embeddings = [embedding1, embedding2, embedding3]
most_similar_index = select_context(question, embeddings)
context = [text1, text2, text3][most_similar_index]

# 執行問答
answer = answer_question(question, context)
print(answer)
