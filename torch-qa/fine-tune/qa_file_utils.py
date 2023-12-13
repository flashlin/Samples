import json
import re
from jinja2 import Environment

"""
Q: How are you
A: Hi
QuestionByPrevTodoList: What is {key}
Template: {% set a=[('a',1), ('b',2)] %}
{% for item in a %}
Q: Who created {{item[0]}}?
A: I use {{item[1]}}
{% endfor %}
:EndTemplate
"""

def create_regex(patterns: list[str]):
    regex_patterns = []
    for pattern in patterns:
        rg_pattern = re.compile(pattern, re.IGNORECASE)
        regex_patterns.append(rg_pattern)
    return regex_patterns


def is_match(txt: str, regex_patterns):
    for pattern in regex_patterns:
        match = pattern.match(txt)
        if match:
            captured_text = match.group(1).strip()
            return captured_text
    return None



class QuestionAnswerContext:

    def __init__(self):
        self.read_state = QuestionAnswerReadyState(self)
        self.question_answer_list = []
        self.questions = []
        self.answers = []
        self.yield_fn = None

    def read_line(self, line: str):
        self.read_state.read_line(line)

    def output_question_answer(self):
        if len(self.answers) > 0:
            self.question_answer_list.append((self.questions, self.answers))
            if self.yield_fn is not None:
                self.yield_fn(self.questions, self.answers)
        self.questions = []
        self.answers = []

    def flush(self):
        self.read_state.flush()


QUESTION_PATTERNS = create_regex([r'Question \d+:(.*)', r'Question:(.*)', r'Q\d+:(.*)', r'Q:(.*)'])
ANSWER_PATTERN = create_regex([r'Answer:(.*)', r'A:(.*)'])
QUESTION_BY_PREV_TODO_LIST_PATTERN = create_regex([r'QuestionByPrevTodoList:(.*)'])
TEMPLATE_PATTERN = create_regex([r'Template:(.*)'])
TEMPLATE_END_PATTERN = create_regex([r':EndTemplate(.*)'])


class QuestionAnswerReadyState:

    def __init__(self, context: QuestionAnswerContext):
        self.context = context

    def read_line(self, line: str):
        captured_text = is_match(line, TEMPLATE_PATTERN)
        if captured_text:
            self.context.read_state = TemplateReadState(self.context, captured_text)
            return
        captured_text = is_match(line, QUESTION_BY_PREV_TODO_LIST_PATTERN)
        if captured_text:
            self.context.read_state = QuestionByPrevTodoListReadState(self.context, captured_text)
            return
        captured_text = is_match(line, QUESTION_PATTERNS)
        if captured_text:
            self.context.read_state = QuestionReadState(self.context, captured_text)

    def flush(self):
        pass


class QuestionReadState:
    def __init__(self, context: QuestionAnswerContext, question: str):
        self.context = context
        self.buffer = question

    def read_line(self, line: str):
        answer_captured_text = is_match(line, ANSWER_PATTERN)
        if answer_captured_text is not None:
            self.flush_buffer()
            self.context.read_state = AnswerReadState(self.context, answer_captured_text)
            return
        question_captured_text = is_match(line, QUESTION_PATTERNS)
        if question_captured_text is not None:
            self.flush_buffer()
            self.buffer = question_captured_text
            return
        self.buffer += '\r\n' + line

    def flush_buffer(self):
        self.context.questions.append(self.buffer.strip())

    def flush(self):
        pass


def split_to_key_value(text: str):
    colon_index = text.find(':')
    if colon_index != -1:
        key = text[1:colon_index].strip()
        value = text[colon_index + 1:].strip()
        return key, value
    return None, None

def parse_markdown_list(markdown_text: str):
    pattern = r'\*.*?(?=\n\*|\Z)'
    matches = re.findall(pattern, markdown_text, re.DOTALL)
    for item in matches:
        key, value = split_to_key_value(item)
        yield key, value


class QuestionByPrevTodoListReadState:
    def __init__(self, context: QuestionAnswerContext, question: str):
        self.context = context
        self.buffer = question

    def read_line(self, line: str):
        captured_text = is_match(line, TEMPLATE_PATTERN)
        if captured_text is not None:
            self.flush_buffer()
            self.context.read_state = TemplateReadState(self.context, captured_text)
            return
        captured_text = is_match(line, QUESTION_PATTERNS)
        if captured_text is not None:
            self.flush_buffer()
            self.context.read_state = QuestionReadState(self.context, captured_text)
            return
        self.buffer += '\r\n' + line

    def flush_buffer(self):
        question_template = self.buffer.strip()
        _, answer_context = self.context.question_answer_list[-1]
        for key, answer in parse_markdown_list(answer_context[0]):
            question = question_template.format(key=key)
            self.context.question_answer_list.append(([question], [answer]))

    def flush(self):
        self.flush_buffer()
        self.context.output_question_answer()
        self.context.read_state = QuestionAnswerReadyState(self.context)


class TemplateReadState:
    def __init__(self, context: QuestionAnswerContext, template: str):
        self.context = context
        self.buffer = template

    def read_line(self, line: str):
        captured_text = is_match(line, TEMPLATE_END_PATTERN)
        if captured_text is not None:
            self.flush_buffer()
            self.context.read_state = QuestionAnswerReadyState(self.context)
            return
        self.buffer += '\r\n' + line

    def flush_buffer(self):
        template_content = self.buffer.strip()
        print(f"{template_content=}")
        env = Environment()
        # env.globals['custom_function'] = custom_function
        # {{ custom_function(3, 5) }}
        template = env.from_string(template_content)
        template_output = template.render()
        print(f"{template_output=}")

        inner_qa = QuestionAnswerContext()
        lines = template_output.splitlines()
        for line in lines:
            line = line.lstrip()
            inner_qa.read_line(line)
        inner_qa.flush()
        for questions, answers in inner_qa.question_answer_list:
            self.context.question_answer_list.append((questions, answers))

    def flush(self):
        self.flush_buffer()
        self.context.output_question_answer()
        self.context.read_state = QuestionAnswerReadyState(self.context)


class AnswerReadState:
    def __init__(self, context: QuestionAnswerContext, answer: str):
        self.context = context
        self.buffer = answer

    def read_line(self, line: str):
        question_by_prev_todo_list = is_match(line, QUESTION_BY_PREV_TODO_LIST_PATTERN)
        if question_by_prev_todo_list is not None:
            self.flush_buffer()
            self.context.read_state = QuestionByPrevTodoListReadState(self.context, question_by_prev_todo_list)
            return
        captured_text = is_match(line, TEMPLATE_PATTERN)
        if captured_text is not None:
            self.flush_buffer()
            self.context.read_state = TemplateReadState(self.context, captured_text)
            return
        question_captured_text = is_match(line, QUESTION_PATTERNS)
        if question_captured_text is not None:
            self.flush_buffer()
            self.context.read_state = QuestionReadState(self.context, question_captured_text)
            return
        answer_captured_text = is_match(line, ANSWER_PATTERN)
        if answer_captured_text is not None:
            self.context.answers.append(self.buffer.strip())
            self.buffer = answer_captured_text
            return
        self.buffer += '\r\n' + line

    def flush_buffer(self):
        self.context.answers.append(self.buffer.strip())
        self.context.output_question_answer()

    def flush(self):
        self.flush_buffer()
        self.context.read_state = QuestionAnswerReadyState(self.context)


def query_qa_file(file: str):
    qa = QuestionAnswerContext()
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            # line = line.strip()
            qa.read_line(line)
        qa.flush()
    for questions, answers in qa.question_answer_list:
        for question in questions:
            for answer in answers:
                yield question.strip(), answer.strip()


def convert_qa_md_file_to_train_jsonl(md_file, jsonl_file, mode:str = "w"):
    with open(jsonl_file, mode, encoding='utf-8') as jfile:
        for question, answer in query_qa_file(md_file):
            json_line = json.dumps({
                'instruction': question,
                'input': '',
                'output': answer,
                'history': []
            })
            jfile.write(json_line+'\r\n')


if __name__ == '__main__':
    file = 'results/test.md'
    for q, a in query_qa_file(file):
        print(f"{q=}")
        print(f"{a=}")
        print("----")


