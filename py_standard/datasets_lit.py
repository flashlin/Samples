import re
from datasets import load_dataset
import shutil


def load_jsonl_dataset(jsonl_files):
    shutil.rmtree('data_cache')
    return load_dataset('json', data_files=jsonl_files, cache_dir='data_cache')


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
