from io_utils import query_sub_files
from langchain_lit import load_markdown_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse
from qa_file_utils import is_match, ANSWER_PATTERN

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
        answer_captured_text = is_match(line, ANSWER_PATTERN)
        if answer_captured_text is not None:
            self.buff += ' '
        self.buff += line

    def flush(self):
        if self.buff == "":
            return
        self.context.paragraph_list.append(self.buff)
        self.buff = ""


def extract_text(folder: str):
    files = query_sub_files(folder, ['.txt', '.md'])
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            yield file, f.read()


def split_text_to_paragraphs(text: str):
    paragraph_parser = ParagraphsContext()
    for line in text.split('\n'):
        paragraph_parser.read_line(line)
    paragraph_parser.flush()
    for paragraph in paragraph_parser.paragraph_list:
        yield paragraph


def extract_paragraphs(folder: str):
    for source, text in extract_text(folder):
        for paragraph in split_text_to_paragraphs(text):
            yield source, paragraph


def first_element(input_iterable):
    iterator = iter(input_iterable)
    return next(iterator)


if __name__ == '__main__':
    args = get_args()
    folder = './data-user'
    with open('./results/paragraphs.txt', 'w', encoding='utf-8') as f:
        for source, paragraph in extract_paragraphs(folder):
            f.write("Question: ???\r\n")
            f.write(f"Answer: {paragraph}\r\n")
            f.write('\r\n\r\n')

    source, first_paragraph = first_element(extract_paragraphs(folder))
    print(f"{first_paragraph=}")
    