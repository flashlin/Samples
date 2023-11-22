#$ pip install autotrain-advanced
#$ autotrain -h
#$ autotrain setup --update-torch
import csv
import re

llama2_train_template = """
###Human:
{question}
###Assistant:
{answer}
"""

def create_patterns(patterns):
    rg_patterns = []
    for pattern in patterns:
        rg_pattern = re.compile(pattern, re.IGNORECASE)
        rg_patterns.append(rg_pattern)
    return rg_patterns


def is_match(txt: str, rg_patterns):
    for pattern in rg_patterns:
        match = pattern.match(txt)
        if match:
            captured_text = match.group(1).strip()
            return captured_text
    return None


def read_qa_txt_file(file: str):
    question_patterns = create_patterns([r'^Question \d+:(.*)', r'^Q\d+:(.*)'])
    answer_patterns = create_patterns([r'^Answer:(.*)', r'^A\d+:(.*)'])
    with open(file, 'r') as file:
        for line in file:
            line = line.strip()
            captured_text = is_match(line, question_patterns)
            if captured_text:
                question = captured_text
            captured_text = is_match(line, answer_patterns)
            if captured_text:
                answer = captured_text
                if question != '' and answer != '':
                    yield question, answer
                question = ""
                answer = ""
            # if line.startswith("---"):
            #     in_answer = False
            #     yield question, answer
            #     answer = ""
            # if in_answer:
            #     answer += line


def convert_qa_txt_file_to_csv(txt_file, csv_file):
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["question", "answer"])
        for question, answer in read_qa_txt_file(txt_file):
            writer.writerow([question, answer])


def convert_qa_txt_file_to_train_csv(txt_file, csv_file):
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text"])
        for question, answer in read_qa_txt_file(txt_file):
            text = llama2_train_template.format(question=question, answer=answer)
            writer.writerow([text])


# convert_qa_txt_file_to_csv("qa.txt", "qa.csv")
convert_qa_txt_file_to_train_csv("qa.txt", "train.csv")

# autotrain llm --train 
# --project_name mistral-7b-finetuned
# --model bn22/Mistral-7B-Instruct-v0.1-sharded 
# --data_path . 
# --use_peft 
# --use_int4 
# --learning_rate 2e-4 
# --train_batch_size 12 
# --num_train_epochs 3 
# --trainer sft 
# --target_modules q_proj,v_proj 
# --push_to_hub 
# --repo_id ashishpatel26/mistral-7b-mj-finetuned


