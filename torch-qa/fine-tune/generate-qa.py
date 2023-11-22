import csv
import os
import re

llama2_train_template = """
###Human:
{question}
###Assistant:
{answer}
"""

def append_to_file(txt, file):
    with open(file, 'a') as f:
        f.write(txt + '\r\n')

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
                answer += captured_text + '\r\n'
                is_answer = True
                continue
            if line.startswith("Answer:"):
                answer = ""
                is_answer = True
                continue
            if line.startswith('---'):
                is_answer = False
                if question != '' and answer != '':
                    yield question, answer
                question = ""
                answer = ""
            if is_answer:
                answer += line + '\r\n'
        if question != '' and answer != '':
            yield question, answer


def convert_qa_md_to_csv(md_file: str, qa_file: str):
    with open(qa_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["question", "answer"])
        for question, answer in query_qa_md(md_file):
            writer.writerow([question, answer])


def list_games(folder):
    filenames = os.listdir(folder)
    game_name_pattern = re.compile(r'(.*)\-rule\-\d+\.md$', re.IGNORECASE)
    for filename in filenames:
        match = game_name_pattern.match(filename)
        if match:
            captured = match.group(1).strip()
            yield captured


def clean_files(folder):
    filenames = os.listdir(folder)
    s1 = "You can find the information about betting rules and game instructions at the"
    remove_files = []
    for filename in filenames:
        with open(f"{folder}/{filename}", 'r', encoding='utf-8') as f:
            content = f.read()
            if s1 in content:
                remove_files.append(filename)
    print(f"{remove_files=}")
    for filename in remove_files:
        os.remove(f'{folder}/{filename}')

if __name__ == '__main__':
    # clean_files("./data")
    convert_qa_md_to_csv("./qa.txt", "./qa.csv")
    # game_names = {}
    # for game_name in list_games('./data'):
    #     game_names[game_name] = True
    # for game_name in game_names.keys():
    #     append_to_file(game_name, "gamename.txt")
