import re
from datasets import load_dataset
import shutil


def load_jsonl_dataset(jsonl_files):
    shutil.rmtree('data_cache')
    return load_dataset('json', data_files=jsonl_files, cache_dir='data_cache')


def eval_in_text(text, func_name, user_fn=None):
    fn_pattern = re.compile("@(" + func_name + r"\(.*\)" +")")
    matches = fn_pattern.finditer(text)
    new_text = text
    for match in matches:
        fn_text = match.group(1)
        if user_fn:
            local_dict = {func_name: user_fn}
            result = eval(fn_text, {}, local_dict)
        else:
            result = eval(fn_text)
        new_text = new_text.replace('@' + fn_text, result)
        return new_text
    return text


def extract_qa_prompt(title=None):
    if title:
        prompt = """The following content pertains to "{title}", Extract information into Q&A, ensuring 
        answers are sourced from the following content."""
        return prompt.format(title=title)
    return """Extract below information into Q&A, ensuring answers are sourced from the following content."""


def should_flush_data(match_label, prev_label):
    if 'Instruction' == match_label and prev_label == 'Answer':
        return True
    if 'Question' == match_label and prev_label == 'Answer':
        return True
    return False


def replace_escape_string(text):
    text = text.replace('@Instruction:', 'Instruction:')
    text = text.replace('@Question:', 'Question:')
    text = text.replace('@Answer:', 'Answer:')
    text = eval_in_text(text, "extract_qa_prompt")
    return text


def flush_qa_data(data):
    if 'Instruction' not in data:
        data['Instruction'] = ''
    data['Instruction'] = replace_escape_string(data['Instruction'])
    data['Question'] = replace_escape_string(data['Question'])
    data['Answer'] = replace_escape_string(data['Answer'])
    return data


def read_pretrained_qa_file(qa_file):
    data = {}
    current_label = None
    labels = ['Instruction', 'Question', 'Answer']
    labels_pattern = '(' + '|'.join(labels) + ')'
    label_pattern = re.compile(r'^' + labels_pattern + r':\s*(.*)')
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            match = label_pattern.match(line)
            if match:
                match_label = match.group(1)
                if should_flush_data(match_label, current_label):
                    if 'Question' in data and 'Answer' in data:
                        yield flush_qa_data(data)
                    data = {}
                current_label = match_label
                data[current_label] = match.group(2)
            elif current_label and line:
                data[current_label] += '\r\n' + line
        if 'Question' in data and 'Answer' in data:
            yield flush_qa_data(data)
