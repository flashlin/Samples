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
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt


documents = SimpleDirectoryReader('data/').load_data()
system_prompt = """You ara a Q&A assistant.
Your goal is to answer question as accurately as possible base on the documents.
"""

query_prompt = SimpleInputPrompt('<|USER|>{query_str}<|ASSISTANT|>')

MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_prompt,
    tokenizer_name=MODEL_NAME,
    model_name=MODEL_NAME,
    device_map='auto',
    model_kwargs={'torch_dtype': torch.float16, "load_in_8bit": True}
)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
)

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()
response = query_engine.query("how to new new b2b2c domain?")

print(response)
