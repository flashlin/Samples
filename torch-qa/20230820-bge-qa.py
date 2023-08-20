#pip -q install langchain huggingface_hub tiktoken
#pip -q install chromadb
#pip -q install PyPDF2 pypdf sentence_transformers
#pip -q install --upgrade together
#pip -q install -U FlagEmbedding
import transformers
from langchain import PromptTemplate, FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import HuggingFacePipeline
from lanchainlit import load_model, load_tokenizer, load_llm, load_chain
from pdf_utils import load_and_split_pdf_texts_from_directory
from vectordb_utils import load_chroma_from_documents

hf_token = ""
device = "cuda"
embedding_model_name = "BAAI/bge-base-en"
model_name = 'anon8231489123/gpt4-x-alpaca-13b-native-4bit-128g'

encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
embedding = HuggingFaceBgeEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs=encode_kwargs
)


persist_directory = 'output/faissdb'
texts = load_and_split_pdf_texts_from_directory('data')


def split_documents_to_vector_db(split_documents, embeddings, db_path):
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    #texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(split_documents, embeddings)
    db.save_local(db_path)
    return db


split_documents_to_vector_db(texts, embedding, persist_directory)

vectordb = load_chroma_from_documents(texts, embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
print("--------------------------------------------")
print("embedding ready")

# llm = CTransformers(model=model_name,
#                     model_type="llama",
#                     max_new_tokens=512,
#                     temperature=0.1)

model = load_model(hf_token, model_name)
tokenizer, stopping_criteria = load_tokenizer(hf_token, model_id=model_name, device=device)
llm = load_llm(model, tokenizer, stopping_criteria)
chain = load_chain(llm, retriever)



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
print('llm ready')

while True:
    prompt = input("query: ")
    if prompt == 'q':
        break
    # prompt = "How to add new b2b2c domain?"
    output = qa_llm({'query': prompt})
    print(output["result"])