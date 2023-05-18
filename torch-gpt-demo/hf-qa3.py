import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from langchain.chains import RetrievalQA
from pdf_utils import splitting_documents_into_texts, load_txt_documents_from_directory
from vectordb_utils import load_chroma_from_documents
from typing import Callable

tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

def get_embed_text(text: str):
    # input_ids = tokenizer.encode(text, max_length=512, truncation=True, return_tensors="pt")
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
    embedding = torch.mean(last_hidden_states, dim=1).numpy()
    return embedding

embedding = Callable(get_embed_text)


documents = load_txt_documents_from_directory('./news')
texts = splitting_documents_into_texts(documents)

# all_embed_text = []
# for doc in texts:
#     metadata = doc.metadata
#     embed_text = get_embed_text(doc.page_content)
#     all_embed_text.append(embed_text)

vectordb = load_chroma_from_documents(texts, embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})


# qa_chain = RetrievalQA.from_chain_type(llm=model,
#                                        chain_type="stuff",
#                                        retriever=retriever,
#                                        return_source_documents=True)

"""
# 創建一個新的數據集
dataset = client.create_dataset("my-dataset")

# 創建一個張量來保存嵌入
embeddings_tensor = dataset.create_tensor("embeddings")

# 假設您已經有了一個嵌入列表
embeddings = [...]

# 將嵌入添加到張量中
embeddings_tensor.extend(embeddings)

# 提交更改
dataset.commit()
"""




# all_metadatas = []
# all_ids = []
# for text, idx in enumerate(documents):
#     all_metadatas.append({
#         "source": "my_source"
#     })
#     all_ids.append(f'id{idx}')
#
#
# # 初始化 Deep Lake 客戶端

# # 獲取數據集
# dataset = client.get_dataset("my-dataset")
# # 獲取嵌入張量
# embeddings_tensor = dataset["embeddings"]
# # 假設您已經有了一個問題文本
# question_text = "What is the capital of France?"
# # 使用 RetrievalQA 從 Deep Lake 中檢索最相關的文檔
# retrieval_qa = RetrievalQA()
# nearest_embedding = retrieval_qa.retrieve([question_text], embeddings_tensor)[0]
#
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
