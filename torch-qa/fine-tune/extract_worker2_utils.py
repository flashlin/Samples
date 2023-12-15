import os
from qa_file_utils import is_match, ANSWER_PATTERN, convert_qa_md_file_to_train_jsonl


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

