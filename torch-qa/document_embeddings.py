from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from lanchainlit import load_documents
from langchain.chains import ConversationalRetrievalChain

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name,
                                   model_kwargs={"device": "cuda"})

web_links = [
    "https://www.databricks.com/",
    "https://help.databricks.com",
]
loader = WebBaseLoader(web_links)
web_documents = loader.load()
data_documents = load_documents('data')
documents = web_documents + data_documents

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)
vectorstore = FAISS.from_documents(all_splits, embeddings)


def get_chat_answer(llm, query, chat_history):
    global embeddings, vectorstore
    chain = ConversationalRetrievalChain.from_llm(llm,
                                                  vectorstore.as_retriever(),
                                                  return_source_documents=True)
    vectorstore = FAISS.from_documents(all_splits, embeddings)
    result = chain({"question": query, "chat_history": chat_history})
    return result['answer']
