from langchain_lit import load_markdown_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Extract Text ")
    parser.add_argument("model_name", nargs='?', help="Name of your model")
    args = parser.parse_args()
    return args


class ParagraphsContext:
    def __init__(self):
        self.read_state = ParagraphReadyState(self)
        self.paragraph_list = []
        self.yield_fn = None

    def read_line(self, line: str):
        self.read_state.read_line(line)

    def output_paragraph(self, paragraph: str):
        self.paragraph_list.append(paragraph)
        if self.yield_fn is not None:
            self.yield_fn(paragraph)

    def flush(self):
        self.read_state.flush()


class ParagraphReadyState:
    def __init__(self, context: ParagraphsContext):
        self.context = context
        self.buff = ""

    def read_line(self, line: str):
        if line.strip() == "":
            self.flush()
            return
        if self.buff != "":
            self.buff += "\n"
        self.buff += line

    def flush(self):
        if self.buff == "":
            return
        self.context.paragraph_list.append(self.buff)
        self.buff = ""



def extract_text(folder: str, chunk_size=1000 * 3):
    documents = load_markdown_documents(folder)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)
    for doc in all_splits:
        content = doc.page_content
        source = doc.metadata['source']
        yield source, content


def split_text_to_paragraphs(text: str):
    paragraph_parser = ParagraphsContext()
    for line in text.split('\n'):
        paragraph_parser.read_line(line)
    paragraph_parser.flush()
    for paragraph in paragraph_parser.paragraph_list:
        yield paragraph


def extract_paragraphs(folder: str, chunk_size=1000 * 3):
    for source, text in extract_text(folder, chunk_size):
        for paragraph in split_text_to_paragraphs(text):
            yield source, paragraph


if __name__ == '__main__':
    args = get_args()
    folder = './data-user'
    for source, paragraph in extract_paragraphs(folder):
        print(f"{paragraph=}")
