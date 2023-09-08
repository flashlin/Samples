from langchain.vectorstores import Chroma
from langchain import FAISS

class MyEmbeddingFunction:
    def __init__(self, embedding_func):
        """
            embedding_function = MyEmbeddingFunction(get_embed_text)
            vectordb = load_chroma_from_documents(texts, embedding_function)
        :param embedding_func:
        """
        self.embedding_func = embedding_func

    def embed_documents(self, documents):
        embeddings = []
        for doc in documents:
            embedding = self.embedding_func(doc)
            embeddings.append(embedding)
        embeddings = [embedding.tolist() for embedding in embeddings]
        return embeddings

    def embed_query(self, query):
        embeddings = self.embedding_func(query)
        embeddings = [embedding.tolist() for embedding in embeddings]
        return embeddings


def load_chroma_from_documents(texts, embedding, persist_directory='./output/db'):
    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embedding,
                                     persist_directory=persist_directory)
    return vectordb


def split_documents_to_vector_db(split_documents, embeddings, db_path):
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    #texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(split_documents, embeddings)
    db.save_local(db_path)
    return db
