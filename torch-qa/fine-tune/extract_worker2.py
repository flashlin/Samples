import os
from io_utils import query_sub_files
from langchain_lit import load_markdown_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse
from llama_cpp import Llama
import time
import re
from finetune_utils import load_finetune_config
from finetune_lit import load_peft_model, ask_llama2_instruction_prompt, get_finetune_model_name
from qa_file_utils import query_qa_file
import shutil
from extract_worker2_utils import ParagraphsContext


MODEL_NAME = "Mistral-7B-Instruct-v0.1"

def get_args():
    parser = argparse.ArgumentParser(description="Extract Text ")
    parser.add_argument("model_name", nargs='?', help="Name of your model")
    parser.add_argument('-w', action='store_true', default=False, help='override output')
    args = parser.parse_args()
    return args

def get_file_content(file: str):
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()


def split_text_to_paragraphs(text: str):
    paragraph_parser = ParagraphsContext()
    for line in text.split('\n'):
        paragraph_parser.read_line(line)
    paragraph_parser.flush()
    for paragraph in paragraph_parser.paragraph_list:
        yield paragraph


def extract_paragraphs_from_file(file: str):
    file_content = get_file_content(file)
    for paragraph in split_text_to_paragraphs(file_content):
        yield file, paragraph

def first_element(input_iterable):
    iterator = iter(input_iterable)
    return next(iterator)


#################
def generate_question(llm, content: str):
    template = get_file_content('./prompt.txt')
    prompt = template.format(content=content)
    answer = llm.ask(prompt)
    answer = answer.strip()
    return answer

def generate_summary(llm, content: str):
    template = get_file_content('./prompt-summary.txt')
    prompt = template.format(content=content)
    answer = llm.ask(prompt)
    return answer


def extract_topics(llm, content: str):
    template = get_file_content('./prompt-topics.txt')
    prompt = template.format(content=content)
    answer = llm.ask(prompt)
    return answer

def generate_qa_from_topics(llm, content, topics):
    template = get_file_content('./prompt-qa-from-topics.txt')
    prompt = template.format(content=content, topics=topics)
    qa_data = llm.ask(prompt)
    return qa_data

def extrac_question_body(question_line: str):
    match = re.match(f'\d+. (.*)', question_line)
    if match:
        return match.group(1).strip()
    return None


def split_questions_content(content: str):
    question_lines = re.findall(r'\d+\. .*', content)
    questions = []
    for question_line in question_lines:
        q = extrac_question_body(question_line)
        questions.append(q)
    return questions


class LLM:
    def __init__(self):
        model, tokenizer, generation_config = self.load_model()
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.device = 'cuda'

    def load_model(self):
        base_model = f"../models/{MODEL_NAME}"
        model, tokenizer = load_peft_model(base_model, None)
        generation_config = model.generation_config
        generation_config.do_sample = True
        generation_config.max_new_tokens = 2048
        generation_config.temperature = 0.7
        generation_config.top_p = 0.75
        generation_config.num_return_sequences = 1
        generation_config.pad_token_id = tokenizer.eos_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id
        return model, tokenizer, generation_config

    def ask(self, user_input: str):
        answer = ask_llama2_instruction_prompt(model=self.model,
                                               generation_config=self.generation_config,
                                               tokenizer=self.tokenizer,
                                               device=self.device,
                                               question=user_input)
        idx = answer.find('[/INST]')
        if idx > 0:
            answer = answer[idx + 7:]
        return answer


def append_qa_data(content: str, llm_qa_file: str):
    with open(llm_qa_file, 'a', encoding='utf-8') as f:
        f.write(content)
        f.write("\n\n\n")


def generate_qa_data(llm, content: str, output_llm_qa_file):
    answer = generate_question(llm, content)
    append_qa_data(answer, output_llm_qa_file)
    print(f"{answer}")
    print("-------------------------------------------------------------------")
    summary = generate_summary(llm, content)
    append_qa_data("# qa summary", output_llm_qa_file)
    answer = generate_question(llm, summary)
    append_qa_data(answer, output_llm_qa_file)
    print(f"{answer}")
    print("-------------------------------------------------------------------")
    topics = extract_topics(llm, content)
    append_qa_data("# qa topic", output_llm_qa_file)
    qa_for_topics = generate_qa_from_topics(llm, content, topics)
    append_qa_data(qa_for_topics, output_llm_qa_file)
    print("\n\n\n")


if __name__ == '__main__':
    args = get_args()

    start_time = time.time()
    llm = LLM()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{MODEL_NAME} Loaded. Take {execution_time} sec.")

    output_llm_qa_file = f"./results/llm-qa.md"
    if args.w:
        if os.path.exists(output_llm_qa_file):
            os.remove(output_llm_qa_file)

    for idx, file in enumerate(query_sub_files('./data', ['.txt', '.md'])):
        content = get_file_content(file)
        print(f"process {file}")
        append_qa_data(f'# {file}', output_llm_qa_file)
        generate_qa_data(llm, content=content, output_llm_qa_file=output_llm_qa_file)
        shutil.move(file, './data-processed/')
