from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema import Document
from langchain.document_loaders import DirectoryLoader
from langchain.storage import InMemoryStore
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from llm_utils import LlmEmbeddings
from qdrant_utils import QdrantVectorStore
from lanchainlit import load_txt_documents, load_markdown_documents
from langchain.retrievers.merger_retriever import MergerRetriever


class RetrievalHelper:
    def __init__(self, vector_db, llm, llm_embeddings):
        self.vector_db = vector_db
        self.llm = llm
        self.llm_embedding = llm_embeddings

    def get_retriever(self, collection_name: str):
        store = self.vector_db.get_store(collection_name)
        return store.as_retriever()

    def get_parent_document_retriever(self, collection_name: str):
        vector_store = self.vector_db.get_store(collection_name)
        store = InMemoryStore()
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        big_chunks_retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        return big_chunks_retriever

    def get_lot_retriever(self, collection_names: list[str]):
        retrievers = []
        for collection_name in collection_names:
            retriever = self.get_parent_document_retriever(collection_name)
            retrievers.append(retriever)
        lot_retriever = MergerRetriever(retrievers=retrievers)
        return lot_retriever

    def add_parent_document(self, collection_name: str, docs: list[Document]):
        big_chunks_retriever = self.get_parent_document_retriever(collection_name)
        big_chunks_retriever.add_documents(docs)

    def get_parent_document_retriever_qa(self, collection_name: str):
        """
            answer = qa.run(query)
        :param collection_name:
        :return:
        """
        big_chunks_retriever = self.get_parent_document_retriever(collection_name)
        qa = RetrievalQA.from_chain_type(llm=self.llm,
                                         chain_type="stuff",
                                         retriever=big_chunks_retriever)
        return qa

    def merge_parent_document_retriever_qa(self, collection_names: list[str]):
        lot_retriever = self.get_lot_retriever(collection_names)

        qa = RetrievalQA.from_chain_type(llm=self.llm,
                                         chain_type="stuff",
                                         retriever=lot_retriever,
                                         chain_type_kwargs=self.create_prompt_kwargs())
        return qa

    def create_prompt_kwargs(self, prompt_template: str = None):
        if prompt_template is None:
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            {context}
            Question: {question}
            Answer in English:"""
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": prompt}
        return chain_type_kwargs


def load_llm_model():
    model_name = "TheBloke_Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    # model_name = "TheBloke_Mistral-7B-OpenOrca-GGUF/mistral-7b-openorca.Q4_K_M.gguf"
    llm = LlamaCpp(
        model_path=f"../models/{model_name}",
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        n_ctx=2048,  # 請求上下文 ValueError: Requested tokens (1130) exceed context window of 512
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,  # True
        streaming=True,
        n_gpu_layers=52,
        n_threads=4,
    )
    return llm


class LlmQaChat:
    def __init__(self, llm, vector_db):
        self.llm = llm
        self.vector_db = vector_db

    def create_llm_qa(self, retriever):
        question_prompt = PromptTemplate.from_template(
            """You are QA Bot. If you don't know the answer, just say that you don't know, don't try to make up an answer.""")
        llm_qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            condense_question_prompt=question_prompt,
            return_source_documents=True,
            verbose=False
        )
        return llm_qa

    def create_retrieval_qa(self, retriever):
        chain_type_kwargs = {
            "verbose": False,
            "prompt": self.create_prompt(),
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"
            )
        }
        llm_qa = RetrievalQA.from_chain_type(llm=self.llm,
                                             chain_type="stuff",
                                             verbose=False,
                                             retriever=retriever,
                                             chain_type_kwargs=chain_type_kwargs)
        return llm_qa

    def create_prompt(self, prompt_template: str = None):
        if prompt_template is None:
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            {context}
            Question: {question}
            Answer in English:"""
        return PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

    def ask(self, question: str):
        llm_qa = self.create_llm_qa(self.vector_db.get_store("sample1").as_retriever())
        return self.ask_question_with_context(llm_qa, question, [])

    def ask_retriever(self, retriever, question: str):
        llm_qa = self.create_retrieval_qa(retriever)
        result = llm_qa({"query": question})
        return result['result']

    def ask_question_with_context(self, llm_qa, question, chat_history):
        result = llm_qa({"question": question, "chat_history": chat_history})
        # print("answer:", result["answer"])
        chat_history = [(question, result["answer"])]
        return result["answer"], chat_history


def main():
    llm_embeddings = LlmEmbeddings()
    resp = llm_embeddings.get_embeddings("How to use C# write HELLO")
    print(f"{len(resp)=}")
    docs1 = load_txt_documents("../data")
    docs2 = load_markdown_documents("../data")

    print("loading llm")
    llm = load_llm_model()
    print("llm done")
    vector_db = QdrantVectorStore(llm_embeddings)

    retriever = RetrievalHelper(vector_db, llm, llm_embeddings)

    # all_collections = vector_db.get_all_collections()
    # print(f"{all_collections=}")

    vector_db.recreate_collection('sample1')
    # vector_db.create_collection('sample2')
    retriever.add_parent_document('sample1', docs1)
    retriever.add_parent_document('sample1', docs2)
    # retriever.add_parent_document('sample2', docs2)
    print(f"add documents done")

    query = "How to convert a B2B2C domain to a B2C domain?"
    llm_qa_chat = LlmQaChat(llm, vector_db)
    # result1 = vector_db.search('sample1', query)
    # print("===")
    # print(f"{result1=}")

    # worked
    # answer, chat_history = llm_qa_chat.ask(query)
    # print(answer)
    print("------")

    t = retriever.get_lot_retriever(['sample1'])
    # answer, chat_history = llm_qa_chat.ask_retriever(query, t)
    # print(f"lot retriever {answer=}")
    print(f"-------------------------------")
    answer = llm_qa_chat.ask_retriever(t, query)
    print("============================")
    print(f"{answer=}")


    # qa = retriever.get_parent_document_retriever_qa('sample1')
    # qa = retriever.merge_parent_document_retriever_qa(['sample1'])
    # print("merge_parent_document_retriever_qa query...")
    # result = qa.run(query)
    # print("")
    # print("")
    # print("---------------")
    # print(f"{result=}")

    # retriever.create_collection('sample')
    # retriever.upsert_docs('sample', docs)
    # result = retriever.search('sample', 'How to create pinia store in vue3?')
    # print(f"{result=}")


if __name__ == '__main__':
    main()
