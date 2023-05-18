from langchain.vectorstores import Chroma


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


def load_chroma_from_documents(texts, embedding, persist_directory='./output/db'):
    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embedding,
                                     persist_directory=persist_directory)
    return vectordb
