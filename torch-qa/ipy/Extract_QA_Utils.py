from langchain.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from py_standard.io_utils import split_file_path

def load_markdown_documents(data_path: str):
    md_loader = DirectoryLoader(data_path, glob='*.md', loader_cls=UnstructuredMarkdownLoader)
    return md_loader.load()

def load_markdown_document(md_file: str):
    md_path, name, _ = split_file_path(md_file)
    md_filename = f"{name}.md"
    md_loader = DirectoryLoader(md_path, glob=md_filename, loader_cls=UnstructuredMarkdownLoader)
    return md_loader.load()
 
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    all_splits = text_splitter.split_documents(docs)
    return all_splits