import requests
import json

from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain import hub
from langchain.llms import Ollama
from llm_utils import LlmEmbedding

url = "http://localhost:11434/api/generate"

headers = {
    'Content-Type': 'application/json',
}

conversation_history = []

def generate_response(prompt):
    conversation_history.append(prompt)

    full_prompt = "\n".join(conversation_history)

    data = {
        "model": "mistral",
        "stream": False,
        "prompt": full_prompt,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        actual_response = data["response"]
        conversation_history.append(actual_response)
        return actual_response
    else:
        print("Error:", response.status_code, response.text)
        return None

def chat():
   while True:
      user_input = input("請輸入字串: ")
      if user_input == "/bye":
         break
      resp = generate_response(user_input)
      print(resp)
   

def qa_docs():
   loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
   data = loader.load()
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
   all_splits = text_splitter.split_documents(data)
   
   llm_embedding = LlmEmbedding("../models/BAA_Ibge-large-en-v1.5")
   vectorstore = Chroma.from_documents(documents=all_splits,
                                       embedding=llm_embedding.embedding)

   ## Retrieve
   # question = "How can Task Decomposition be done?"
   # docs = vectorstore.similarity_search(question)                                       
   QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

   ollama = Ollama(base_url='http://localhost:11434', model="mistral")
   qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
   while True:
      user_input = input("請輸入字串: ")
      if user_input == "/bye":
         break
      resp = qachain({"query": user_input})
      print(resp['result'])

qa_docs()      
   