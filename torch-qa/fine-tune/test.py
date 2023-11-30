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
    questions = QuestionAnswerLines('Q', [r'Question \d+:(.*)', r'Question:(.*)', r'Q\d+:(.*)', r'Q:(.*)'])
    answers = QuestionAnswerLines('A', [r'Answer:(.*)', r'A:(.*)'])

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            print(f"{line=}")
            first_question = questions.read_line(line)
            first_answer = answers.read_line(line)
            print(f"{questions.buffers=}")
            print(f"{answers.buffers=}")
            if first_answer:
                questions.buffers = questions.buffers[:-1]
                questions.is_reading = False
            if questions.is_reading:
                answers.is_reading = False
            if answers.is_reading:
                questions.is_reading = False
            if first_question and answers.has():
                print("FIRST")
                questions_list = questions.get_list()
                answers_list = answers.get_list()
                print(f"{questions_list=}")
                print(f"{answers_list=}")
                for question in questions_list:
                    for answer in answers_list:
                        yield question, answer
                continue


if __name__ == '__main__':
    file = 'data-user/test.txt'
    for q, a in query_qa_file(file):
        q = q.replace('\r\n', '\\r\\n')
        print(f"Q:{q} A:{a}")

