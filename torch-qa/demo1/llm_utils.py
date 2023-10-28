from langchain.embeddings import HuggingFaceBgeEmbeddings


class LlmEmbeddings:
    def __init__(self):
        # model_name = "BAAI/bge-large-en"
        # encode_kwargs = {'normalize_embeddings': False}
        # hf = HuggingFaceBgeEmbeddings(
        #     model_name=model_name,
        #     encode_kwargs=encode_kwargs
        # )
        """
        import chromadb
from langchain.vectorstores import Chroma
from langchain.document_transformers import (
    LongContextReorder,
)
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.merger_retriever import MergerRetriever
        """
        embedding_model_name = "../models/BAAI_bge-base-en"
        # embedding_model_name = "../models/BAA_Ibge-large-en-v1.5"
        encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
        self.embedding = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs=encode_kwargs
        )

    def get_embeddings(self, text: str) -> list[float]:
        return self.embedding.embed_query(text)
