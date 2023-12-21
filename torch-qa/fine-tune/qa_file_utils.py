import json
import os
import re
from jinja2 import Environment
from itertools import combinations

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

    def __init__(self, is_single: bool = False):
        self.read_state = QuestionAnswerReadyState(self)
        if is_single:
            self.read_state = SingleQuestionAnswerReadyState(self)
        self.question_answer_list = []
        self.questions = []
        self.answers = []
        self.yield_fn = None

    def read_line(self, line: str):
        if line.startswith('#'):
            return
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


class SingleQuestionAnswerReadyState:

    def __init__(self, context: QuestionAnswerContext):
        self.context = context

    def read_line(self, line: str):
        captured_text = is_match(line, QUESTION_PATTERNS)
        if captured_text:
            self.context.read_state = SingleQuestionReadState(self.context, captured_text)

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



class SingleQuestionReadState:
    def __init__(self, context: QuestionAnswerContext, question: str):
        self.context = context
        self.buffer = question

    def read_line(self, line: str):
        answer_captured_text = is_match(line, ANSWER_PATTERN)
        if answer_captured_text is not None:
            self.flush_buffer()
            self.context.read_state = SingleAnswerReadState(self.context, answer_captured_text)
            return
        question_captured_text = is_match(line, QUESTION_PATTERNS)
        if question_captured_text is not None:
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
        self.questions = []

    def read_line(self, line: str):
        captured_text = is_match(line, QUESTION_BY_PREV_TODO_LIST_PATTERN)
        if captured_text is not None:
            self.questions.append(self.buffer.strip())
            self.buffer = captured_text
            return
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
        _, answer_context = self.context.question_answer_list[-1]
        self.questions.append(self.buffer.strip())
        for question_template in self.questions:
            for key, answer in parse_markdown_list(answer_context[0]):
                question = question_template.format(key=key)
                self.context.question_answer_list.append(([question], [answer]))

    def flush(self):
        self.flush_buffer()
        self.context.output_question_answer()
        self.context.read_state = QuestionAnswerReadyState(self.context)


def compute_fn(expr: str):
    return eval(expr)

def mul_fn(a1, b1):
    return a1 * b1

def baccarat_card_value_fn(cards: list[str]) -> int:
    cards_dict = {
        'a': 1,
        'ace': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        'j': 0,
        'q': 0,
        'k': 0,
        'jack': 0,
        'queen': 0,
        'king': 0,
    }
    value = 0
    for card in cards:
        card = card.lower()
        if card not in cards_dict:
            continue
        value += cards_dict[card]
    return value % 10

def who_win_fn(name1, point1, name2, point2, win_message, tie_message):
    if point1 > point2:
        return win_message.format(name=name1)
    if point1 < point2:
        return win_message.format(name=name2)
    return tie_message

def dragon_baccarat_win_odds_fn(bet_name, name1, point1, name2, point2, win_message, tie_message, lose_message):
    winner = who_win_fn(name1, point1, name2, point2, "{name}", tie_message)
    difference = abs(point1 - point2)
    if point1 == 8 or point1 == 9:
        if difference != 0:
            odds = "1 to 1"
            result = "Natural Win"
            return win_message.format(winner=winner, result=result, odds=odds)
        if difference == 0:
            odds = "Push"
            result = "Natural Tie"
            return tie_message.format(result=result, odds=odds)
    if difference == 9:
        odds = "1 to 30"
        result = "9 points difference"
        return win_message.format(winner=winner, result=result, odds=odds)
    if difference == 8:
        odds = "1 to 10"
        result = "8 points difference"
        return win_message.format(winner=winner, result=result, odds=odds)
    if difference == 7:
        odds = "1 to 6"
        result = "7 points difference"
        return win_message.format(winner=winner, result=result, odds=odds)
    if difference == 6:
        odds = "1 to 4"
        result = "6 points difference"
        return win_message.format(winner=winner, result=result, odds=odds)
    if difference == 5:
        odds = "1 to 2"
        result = "5 points difference"
        return win_message.format(winner=winner, result=result, odds=odds)
    if difference == 4:
        odds = "1 to 1"
        result = "4 points difference"
        return win_message.format(winner=winner, result=result, odds=odds)
    return lose_message

def list_to_combinations_dict(a_list):
    keys = [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    if len(keys) < len(a_list):
        raise ValueError(f"list size > keys")
    a_dict = {}
    for value in a_list:
        key = keys.pop(0)
        a_dict[key] = value
    return a_dict

def combinations_fn(items: list[str], n: int):
    combinations_dict = list_to_combinations_dict(items)
    keys = "".join(key for key in combinations_dict.keys())
    results = list(combinations(keys,n))
    new_results = []
    for items in results:
        new_list = []
        for elem in list(items):
            value = combinations_dict[elem]
            new_list.append(value)
        new_results.append(new_list)
    return new_results

def render_template(template_content: str):
    env = Environment()
    env.globals['compute'] = compute_fn
    env.globals['mul'] = mul_fn
    env.globals['baccarat_card_value'] = baccarat_card_value_fn
    env.globals['who_win'] = who_win_fn
    # {{ custom_function(3, 5) }}
    try:
        template = env.from_string(template_content)
        template_output = template.render()
    except:
        raise ValueError(f"ERROR: {template_content}")
    return template_output


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
        template_output = render_template(template_content)
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



class SingleAnswerReadState:
    def __init__(self, context: QuestionAnswerContext, answer: str):
        self.context = context
        self.buffer = answer

    def read_line(self, line: str):
        question_captured_text = is_match(line, QUESTION_PATTERNS)
        if question_captured_text is not None:
            self.flush_buffer()
            self.context.read_state = SingleQuestionReadState(self.context, question_captured_text)
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


def query_qa_file(file: str, is_single: bool=False):
    if os.path.exists(file) is False:
        return
    qa = QuestionAnswerContext(is_single=is_single)
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
    file = 'results/llm-qa.md'
    question_count = 0
    for q, a in query_qa_file(file, True):
        question_count += 1
        print(f"{q=}")
        print(f"{a=}")
        print("----")
    print(f"{question_count=}")


