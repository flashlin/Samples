from langchain_lit import load_markdown_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Extract Text ")
    parser.add_argument("model_name", nargs='?', help="Name of your model")
    args = parser.parse_args()
    return args


def extract_text(folder: str, chunk_size=1000 * 3):
    documents = load_markdown_documents(folder)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)
    for doc in all_splits:
        content = doc.page_content
        source = doc.metadata['source']
        yield source, content


def split_text_to_paragraphs(text: str):
    for line in text.split('\n'):
        yield line


def extract_paragraphs(folder: str, chunk_size=1000 * 3):
    for source, text in extract_text(folder, chunk_size):
        for paragraph in split_text_to_paragraphs(text):
            yield source, paragraph


if __name__ == '__main__':
    args = get_args()
    folder = './data-user'
    for source, paragraph in extract_paragraphs(folder):
        print(f"{paragraph=}")
