import csv
import json
import os
import re
from finetune_lit import create_llama2_finetune_prompt
from io_utils import query_sub_files, split_file_path
from qa_file_utils import query_qa_file, convert_qa_md_file_to_train_jsonl

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
    for question, answer in query_qa_file(file):
        yield question, answer
    # with open(file, 'r', encoding='utf-8') as f:
    #     is_question = False
    #     is_answer = False
    #     question = ""
    #     answer = ""
    #     for line in f:
    #         line = line.strip()
    #         captured_text = is_match(line, create_regex([r'Question ?\d+:(.*)', r'Question:(.*)', r'Q\d+:(.*)']))
    #         if captured_text:
    #             is_question = True
    #             if question != '' and answer != '':
    #                 yield question, answer
    #             question = captured_text
    #             answer = ""
    #             is_answer = False
    #             continue
    #         captured_text = is_match(line, create_regex([r'Answer:(.*)']))
    #         if captured_text:
    #             answer += captured_text + '\r\n'
    #             is_answer = True
    #             continue
    #         if line.startswith("Answer:"):
    #             answer = ""
    #             is_answer = True
    #             continue
    #         if line.startswith('---'):
    #             is_answer = False
    #             if question != '' and answer != '':
    #                 yield question, answer
    #             question = ""
    #             answer = ""
    #         if is_question:
    #             question += '\r\n' + line
    #         if is_answer:
    #             answer += line + '\r\n'
    #     if question != '' and answer != '':
    #         yield question, answer


def convert_qa_md_to_csv(md_file: str, qa_file: str):
    with open(qa_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["question", "answer"])
        for question, answer in query_qa_md(md_file):
            writer.writerow([question, answer])



def convert_train_jsonl_to_json(jsonl_file):
    folder, filename, _ = split_file_path(jsonl_file)
    json_file = f"{folder}/{filename}.json"
    with open(jsonl_file, 'r', encoding='utf-8') as jsonl_reader:
        num_lines = sum(1 for _ in jsonl_reader)
    with open(jsonl_file, 'r', encoding='utf-8') as jsonl_reader:
        with open(json_file, 'w', encoding='utf-8') as json_writer:
            json_writer.write('[\r\n')
            for i, line in enumerate(jsonl_reader):
                json_writer.write(line.strip())
                if i < num_lines-1:
                    json_writer.write(",\r\n")
                else:
                    json_writer.write("\r\n")
            json_writer.write(']\r\n')


def convert_train_jsonl_to_csv(jsonl_file):
    folder, filename, _ = split_file_path(jsonl_file)
    csv_file = f"{folder}/{filename}.csv"
    print(f"{csv_file=}")
    with open(jsonl_file, 'r', encoding='utf-8') as jsonl_reader:
        num_lines = sum(1 for _ in jsonl_reader)
    print(f"{num_lines=}")
    with open(jsonl_file, 'r', encoding='utf-8') as jsonl_reader:
        qa_csv = QaCsv(csv_file)
        qa_csv.renew()
        for i, line in enumerate(jsonl_reader):
            row = json.loads(line)
            qa_csv.write_data(row['instruction'], row['output'])


def create_llama2_instruction_prompt(question, user_input, answer):
    instruction_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:
{output}"""
    return instruction_prompt_template.format(instruction=question, input=user_input, output=answer)


def create_finetune_prompt(question, answer):
    return create_llama2_finetune_prompt(question, answer)
    # return create_orca2_finetune_prompt(question, answer)


def append_qa_to_train_csv_file(train_file: str, question: str, answer: str):
    with open(train_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        prompt = create_finetune_prompt(question, answer)
        csv_writer.writerow([prompt])


class QaCsv:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file

    def write_data(self, question: str, answer: str):
        with open(self.csv_file, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([question, answer])
            csvfile.flush()

    def renew(self):
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])


def append_to_qa_csv_file(md_file: str, csv_file: QaCsv):
    for question, answer in query_qa_md(md_file):
        question = question.strip()
        answer = answer.strip()
        csv_file.write_data(question, answer)


def convert_llm_qa_md_file_to_train_csv(llm_qa_file, train_file):
    if not os.path.exists(llm_qa_file):
        return
    for question, answer in query_qa_md(llm_qa_file):
        question = question.strip()
        answer = answer.strip()


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
    qa_jsonl = "./results/qa.jsonl"
    for idx, file in enumerate(query_sub_files('./data-user', ['.txt', '.md'])):
        if idx == 0:
            convert_qa_md_file_to_train_jsonl(file, qa_jsonl)
        else:
            convert_qa_md_file_to_train_jsonl(file, qa_jsonl, 'a')
    llm_qa_data = './results/llm-qa.md'
    convert_qa_md_file_to_train_jsonl(llm_qa_data, qa_jsonl, 'a')

    convert_train_jsonl_to_json(qa_jsonl)

    #convert_qa_md_file_to_train_jsonl("./data-user/baccarat-cards.md", qa_jsonl)
    convert_train_jsonl_to_csv(qa_jsonl)

