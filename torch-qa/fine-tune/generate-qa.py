import csv
import re

llama2_train_template = """
###Human:
{question}
###Assistant:
{answer}
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


def query_qa_md(file: str):
    with open(file, 'r', encoding='utf-8') as f:
        is_answer = False
        question = ""
        answer = ""
        for line in f:
            line = line.strip()
            captured_text = is_match(line, create_regex([r'Question ?\d+:(.*)', r'Q\d+:(.*)']))
            if captured_text:
                if question != '' and answer != '':
                    yield question, answer
                question = captured_text
                answer = ""
                is_answer = False
            captured_text = is_match(line, create_regex([r'Answer:(.*)']))
            if captured_text:
                is_answer = True
            if line.startswith('---'):
                is_answer = False
                if question != '' and answer != '':
                    yield question, answer
                question = ""
                answer = ""
            if is_answer:
                answer += line + '\r\n'


def convert_qa_md_to_csv(md_file: str, qa_file: str):
    with open(qa_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["question", "answer"])
        for question, answer in query_qa_md(md_file):
            writer.writerow([question, answer])


if __name__ == '__main__':
    convert_qa_md_to_csv("./qa.txt", "./qa.csv")