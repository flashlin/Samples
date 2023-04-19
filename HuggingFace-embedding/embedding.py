# Document Loader
from langchain.document_loaders import TextLoader
import numpy as np
loader = TextLoader('./train.txt', encoding='utf-8')
documents = loader.load()

print(documents)


import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')
    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

#print(wrap_text_preserve_newlines(str(documents[0])))


# Text Splitter
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print(f"docs len={len(docs)}")


# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()

from langchain.vectorstores import FAISS
db = FAISS.from_documents(docs, embeddings)

# sub_dbs = db.split(nb_parts=10, by='ntotal')
# sub_dbs = db.split(nb_parts=10, by='similarity')  # ntotal/nhashbits

#query = "What did the president say about the Supreme Court"
query = "What did the God"
docs = db.similarity_search(query)
print('------------------')
print(query)
print(wrap_text_preserve_newlines(str(docs[0].page_content)))
print('------------------')


def query_sub_dbs():
    # 要進行檢索的查詢向量
    query = "What did the God"
    query_vec = embeddings.embed(query)

    # 在每個小存儲庫中進行檢索
    results = []
    for sub_db in sub_dbs:
        D, I = sub_db.search(np.array([query_vec]), k=5)
        results.append((D, I))

    # 合併檢索結果
    D, I = zip(*results)
    D = np.concatenate(D, axis=1)
    I = np.concatenate(I, axis=1)

    # 根據索引取得相關文檔並打印
    for i in I[0]:
        doc = sub_dbs[i].documents[i]
        print(wrap_text_preserve_newlines(str(doc.page_content)))


#Create QA Chain
# from langchain.chains.question_answering import load_qa_chain
# from langchain import HuggingFaceHub
# llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":512})
# chain = load_qa_chain(llm, chain_type="stuff")




# from langchain.document_loaders import UnstructuredPDFLoader
# from langchain.indexes import VectorstoreIndexCreator
# loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
# index = VectorstoreIndexCreator(
#     embedding=HuggingFaceEmbeddings(),
#     text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders(loaders)

# llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":512})
