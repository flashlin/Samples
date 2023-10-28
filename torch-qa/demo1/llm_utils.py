from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseRetriever
from langchain.schema.language_model import BaseLanguageModel


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


class ConversationalRetrievalChainAgent:
    def __init__(self, llm: BaseLanguageModel, retriever: BaseRetriever):
        question_prompt = PromptTemplate.from_template(
            """You are QA Bot. If you don't know the answer, just say that you don't know, don't try to make up an answer.""")
        self.llm_qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            condense_question_prompt=question_prompt,
            return_source_documents=True,
            verbose=False
        )

    def ask(self, question: str):
        history = []
        result = self.llm_qa({"question": question, "chat_history": history})
        history = [(question, result["answer"])]
        return result["answer"]


class RetrievalQAAgent:
    """
    db.get_store() + RetrievalQAAgent
    不知道為什麼找不到 document
    """
    def __init__(self, llm, retriever):
        chain_type_kwargs = {
            "verbose": False,
            "prompt": self.create_prompt(),
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"
            )
        }
        self.llm_qa = RetrievalQA.from_chain_type(llm=llm,
                                                  chain_type="stuff",
                                                  verbose=False,
                                                  retriever=retriever,
                                                  chain_type_kwargs=chain_type_kwargs)

    def ask(self, question: str):
        result = self.llm_qa({"query": question})
        return result["answer"]

    def create_prompt(self, prompt_template: str = None):
        if prompt_template is None:
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            You are QA Bot. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            {context}
            Question: {question}
            Only return the helpful answer below and nothing else.
            Helpful answer:"""
        return PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
