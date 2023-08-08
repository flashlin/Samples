import asyncio

import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import textwrap
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DB_FAISS_PATH = "models/db_faiss"
MODEL_BIN_NAME = "models/llama-2-7b-chat.ggmlv3.q8_0.bin"


def create_documents_db():
    loader = DirectoryLoader("data/", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(texts, embeddings)
    db.save_local("faiss")
    return db


llm = CTransformers(model=MODEL_BIN_NAME,
                    model_type='llama',
                    config={
                        'max_new_tokens': 256,
                        'temperature': 0.01
                    })


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})
db = create_documents_db()
db = FAISS.load_local("faiss", embeddings)

retriever = db.as_retriever(search_kwargs={'k': 2})

template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""
prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question'])
qa_llm = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type='stuff',
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': prompt})

# ask the AI chat about information in our local files
prompt = "How to add new b2b2c domain?"
output = qa_llm({'query': prompt})
print(output["result"])
