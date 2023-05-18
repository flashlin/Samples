import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from pdf_utils import splitting_documents_into_texts, load_txt_documents_from_directory
from vectordb_utils import load_chroma_from_documents, MyEmbeddingFunction

model_name = "deepset/bert-base-cased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)


def get_embed_text(text: str):
    # input_ids = tokenizer.encode(text, max_length=512, truncation=True, return_tensors="pt")
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
    embedding = torch.mean(last_hidden_states, dim=1).numpy()
    return embedding


documents = load_txt_documents_from_directory('./news')
texts = splitting_documents_into_texts(documents)

embedding_function = MyEmbeddingFunction(get_embed_text)
vectordb = load_chroma_from_documents(texts, embedding_function)
retriever = vectordb.as_retriever(search_kwargs={"k": 10})


def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    score = outputs[1][0, answer_start].item()
    return answer, score

question = "WSL2 is very slow, How to resolve it?"
docs = retriever.get_relevant_documents(question)
answer = ''
best_score = 0
for doc in docs:
    answer, score = answer_question(question, doc.page_content)
    if score > best_score:
        best_answer = answer
        best_score = score
    print(f'{doc.page_content=} {score=}')

print(f'最佳回答: {best_answer}')

# # 使用預先訓練的 BERT 模型回答問題
# # input_ids = tokenizer.encode(question_text, nearest_embedding, return_tensors="pt")
# # with torch.no_grad():
# #     outputs = model(input_ids)
# #     start_logits, end_logits = outputs.start_logits, outputs.end_logits
# #     start_index = torch.argmax(start_logits)
# #     end_index = torch.argmax(end_logits) + 1
# #     answer_text = tokenizer.decode(input_ids[0][start_index:end_index])
#
# def answer_question(question, context):
#     inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
#     input_ids = inputs["input_ids"].tolist()[0]
#     outputs = model(**inputs)
#     answer_start = torch.argmax(outputs.start_logits)
#     answer_end = torch.argmax(outputs.end_logits) + 1
#     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
#     return answer
#
#
# context = "MingMing's birth is 2023-03-01. SmallPig have brother, his name is Flash. Flash and wife life "
# question = "what name is the SmallPig's sister?"
# answer = answer_question(question, context)
# print(f'{question} {answer=}')
