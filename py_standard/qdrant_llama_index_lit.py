import qdrant_client
from llama_index import StorageContext, GPTVectorStoreIndex, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# vector_store = QdrantVectorStore()

class VectorStoreGpt:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.vector_store = QdrantVectorStore(client=client,
                                              collection_name=collection_name)
        self.index = self.create_query_index()

    def create_query_index(self):
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        return GPTVectorStoreIndex.from_documents([], storage_context=storage_context)

    def add_document(self, docs):
        for document in docs:
            self.index.update(document)

    def ask(self, query: str):
        query_engine = self.index.as_query_engine()
        return query_engine.query(query)

