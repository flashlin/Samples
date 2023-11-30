import re

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


class State:
    def read_line(self, line: str) -> bool:
        pass

class QuestionAnswerContext:
    question_answer_list = []

    def __init__(self):
        self.read_state = QuestionAnswerReadyState(self)
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


class QuestionAnswerReadyState:

    def __init__(self, context: QuestionAnswerContext):
        self.context = context

    def read_line(self, line: str):
        captured_text = is_match(line, QUESTION_PATTERNS)
        if captured_text:
            self.context.read_state = QuestionHeadState(self.context, captured_text)

    def flush(self):
        pass


class QuestionHeadState:
    def __init__(self, context: QuestionAnswerContext, question: str):
        self.context = context
        self.buffer = question

    def read_line(self, line: str):
        answer_captured_text = is_match(line, ANSWER_PATTERN)
        if answer_captured_text:
            self.context.questions.append(self.buffer.strip())
            self.context.read_state = AnswerHeadState(self.context, answer_captured_text)
            return
        question_captured_text = is_match(line, QUESTION_PATTERNS)
        if question_captured_text:
            self.context.questions.append(self.buffer.strip())
            self.buffer = question_captured_text
            return
        self.buffer += '\r\n' + line

    def flush(self):
        pass


class AnswerHeadState:
    def __init__(self, context: QuestionAnswerContext, answer: str):
        self.context = context
        self.buffer = answer

    def read_line(self, line: str):
        question_captured_text = is_match(line, QUESTION_PATTERNS)
        if question_captured_text:
            self.context.answers.append(self.buffer.strip())
            self.context.output_question_answer()
            self.context.read_state = QuestionHeadState(self.context, question_captured_text)
            return
        answer_captured_text = is_match(line, ANSWER_PATTERN)
        if answer_captured_text:
            self.context.answers.append(self.buffer.strip())
            self.buffer = answer_captured_text
            return
        self.buffer += '\r\n' + line

    def flush(self):
        self.context.answers.append(self.buffer.strip())
        self.context.output_question_answer()
        self.context.read_state = QuestionAnswerReadyState(self.context)


class QuestionAnswerLines:
    data_list = []
    buffers = []
    is_reading = False

    def __init__(self, name: str, patterns: list[str]):
        self.name = name
        self.pattern = create_regex(patterns)

    def read_line(self, line: str):
        captured_text = is_match(line, self.pattern)
        if captured_text:
            if self.is_reading:
                buffer = '\r\n'.join(self.buffers).strip()
                self.data_list.append(buffer)
                self.buffers = [captured_text]
                return False
            self.is_reading = True
            self.buffers = [captured_text]
            return len(self.data_list) == 0
        if self.is_reading:
            self.buffers.append(line)
            return len(self.data_list) == 0
        return False

    def get_list(self):
        if len(self.buffers) > 0:
            buffer = '\r\n'.join(self.buffers).strip()
            self.data_list.append(buffer)
            self.buffers = []
        data_list = self.data_list
        self.data_list = []
        return data_list

    def has(self):
        if len(self.buffers) > 0:
            return True
        return len(self.data_list) > 0


def query_qa_file(file: str):
    qa = QuestionAnswerContext()
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            qa.read_line(line)
        qa.flush()
    for questions, answers in qa.question_answer_list:
        for question in questions:
            for answer in answers:
                question = question.replace('\r\n', '\\n')
                answer = answer.replace('\r\n', '\\n')
                print(f"{question=} {answer=}")


if __name__ == '__main__':
    file = 'data-user/test.txt'
    query_qa_file(file)


