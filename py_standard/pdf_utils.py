from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# loader = PyPDFLoader("./attention_is_all_you_need.pdf")
# pages = loader.load_and_split()

def load_pdf_documents_from_directory(directory_path):
    loader = DirectoryLoader(directory_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def splitting_documents_into_texts(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts


def load_and_split_pdf_texts_from_directory(directory_path):
    documents = load_pdf_documents_from_directory(directory_path)
    return splitting_documents_into_texts(documents)
