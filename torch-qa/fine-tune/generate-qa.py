import csv
import json
import os
import re
from finetune_lit import create_llama2_finetune_prompt
from qa_file_utils import query_qa_file

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


def convert_qa_md_file_to_train_jsonl(md_file, jsonl_file, mode:str = "w"):
    with open(jsonl_file, mode, encoding='utf-8') as jfile:
        for question, answer in query_qa_md(md_file):
            json_line = json.dumps({
                'instruction': question,
                'input': '',
                'output': answer,
                'history': []
            })
            jfile.write(json_line+'\r\n')


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


def convert_qa_md_file_to_train_csv(md_file, train_file):
    with open(train_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text"])
        for question, answer in query_qa_md(md_file):
            question = question.strip()
            answer = answer.strip()
            text = create_finetune_prompt(question=question, answer=answer)
            writer.writerow([text])


def convert_llm_qa_md_file_to_train_csv(llm_qa_file, train_file):
    if not os.path.exists(llm_qa_file):
        return
    for question, answer in query_qa_md(llm_qa_file):
        question = question.strip()
        answer = answer.strip()
        append_qa_to_train_csv_file(train_file, question, answer)


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
    user_data = "./data-user/qa.txt"
    llm_qa_data = './results/llm-qa.md'
    # clean_files("./data")
    # convert_qa_md_to_csv(user_data, "./results/qa.csv")
    convert_qa_md_file_to_train_jsonl(user_data, "./results/qa.json")
    convert_qa_md_file_to_train_jsonl(llm_qa_data, "./results/qa.json", 'a')
    convert_qa_md_file_to_train_csv(user_data, './results/train.csv')
    convert_llm_qa_md_file_to_train_csv(llm_qa_data, './results/train.csv')
    # game_names = {}
    # for game_name in list_games('./data'):
    #     game_names[game_name] = True
    # for game_name in game_names.keys():
    #     append_to_file(game_name, "gamename.txt")
