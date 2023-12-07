from langchain_lit import load_markdown_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter

def yield_extract_text(folder: str, chunk_size=1000*3):
    documents = load_markdown_documents(folder)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)
    for doc in all_splits:
        content = doc.page_content
        source = doc.metadata['source']
        yield source, content

