# pip -q install langchain openai tiktoken chromadb 
# pip show langchain
# !wget -q https://www.dropbox.com/s/vs6ocyvpzzncvwh/new_articles.zip
# !unzip -q new_articles.zip -d new_articles

import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

os.environ["OPENAI_API_KEY"] = ""

loader = DirectoryLoader('./new_articles/', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()

#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)


# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'
## here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)
# persiste the db to disk
vectordb.persist()
vectordb = None
# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

retriever = vectordb.as_retriever()
docs = retriever.get_relevant_documents("How much money did Pando raise?")
retriever = vectordb.as_retriever(search_kwargs={"k": 2})
retriever.search_type
retriever.search_kwargs


# create the chain to answer questions 
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)
## Cite sources
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
        
# full example
query = "How much money did Pando raise?"
llm_response = qa_chain(query)
process_llm_response(llm_response)


# break it down
query = "What is the news about Pando?"
llm_response = qa_chain(query)
# process_llm_response(llm_response)
llm_response



query = "Who led the round in Pando?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

print(qa_chain.combine_documents_chain.llm_chain.prompt.template)
        