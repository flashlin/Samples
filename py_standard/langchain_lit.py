from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema import BaseRetriever, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.vectorstores import FAISS
from langchain.schema.vectorstore import VectorStore


def load_llm_model_with_callback_handler(model_name: str, callback_handler=None):
    if callback_handler is None:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    else:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler(), callback_handler])
    return LlamaCpp(
        model_path=model_name,
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        n_ctx=2048,
        callback_manager=callback_manager,
        verbose=False,
        streaming=True,
    )


def load_llm_model(model_name: str, display_callback_handler=None):
    return load_llm_model_with_callback_handler(model_name, StreamDisplayHandler(display_callback_handler))


class LlmBot:
    def __init__(self, model_name: str):
        self.callback_handler = StreamDisplayHandler(None)
        self.llm = load_llm_model_with_callback_handler(model_name, self.callback_handler)
        self.chat_history = []

    def ask(self, query: str, display_callback_handler=None) -> str:
        self.callback_handler.container = display_callback_handler
        result = self.llm(query)
        return result


def load_txt_documents(txt_path: str):
    txt_loader = DirectoryLoader(txt_path, glob='*.txt', loader_cls=TextLoader)
    return txt_loader.load()


def load_markdown_documents(data_path: str):
    md_loader = DirectoryLoader(data_path, glob='*.md', loader_cls=UnstructuredMarkdownLoader)
    return md_loader.load()


class LlmEmbedding:
    def __init__(self, model_name: str):
        encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
        self.embedding = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs=encode_kwargs
        )

    def get_embeddings(self, text: str) -> list[float]:
        return self.embedding.embed_query(text)


class ConversationalRetrievalChainAgent:
    history = []

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
        result = self.llm_qa({"question": question, "chat_history": self.history})
        self.history = [(question, result["answer"])]
        return result["answer"]


def create_parent_document_retriever(vector_store: VectorStore):
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



class FaissRetrieval:
    def __init__(self, llm_embedding: LlmEmbedding):
        self.llm_embedding = llm_embedding

    def get_retriever(self, docs):
        vector_store = FAISS.from_documents(docs, self.llm_embedding.embedding)
        return create_parent_document_retriever(vector_store)


class Retrieval:
    def __init__(self, vector_db, llm, llm_embedding: LlmEmbedding):
        self.vector_db = vector_db
        self.llm = llm
        self.llm_embedding = llm_embedding

    def get_retriever(self, collection_name: str):
        store = self.vector_db.get_store(collection_name)
        return store.as_retriever(search_kwargs={"k": 5})

    def get_parent_document_retriever(self, collection_name: str):
        vector_store = self.vector_db.get_store(collection_name)
        return create_parent_document_retriever(vector_store)

    def add_parent_document(self, collection_name: str, docs: list[Document]):
        big_chunks_retriever = self.get_parent_document_retriever(collection_name)
        big_chunks_retriever.add_documents(docs)


class StreamDisplayHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='display'):
        """
        :param container: class
        :param initial_text:
        :param display_method: 'display'
        class MyContainer:
            def display(self, text):
               pass
        """
        self.container = container
        self.display_method = display_method
        self.text = initial_text
        self.new_sentence = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.new_sentence += token
        # self.call_display_func(self.text + "▌")
        self.call_display_func(token)

    def on_llm_end(self, response, **kwargs) -> None:
        self.text = ""
        if self.container is None:
            return
        display_end_function = getattr(self.container, f"{self.display_method}_end", None)
        if display_end_function is not None:
            display_end_function()
        else:
            raise ValueError(f"Invalid display_end_method: {display_end_function}")

    def call_display_func(self, text: str):
        if self.container is None:
            return
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(text)
        else:
            raise ValueError(f"Invalid display_method: {display_function}")



