import ast
import os
import re

import torch
from torch import nn
from torch.utils.data import Dataset

from network import MultiHeadAttention, alist_to_chunks, pad_chunks_list
from tsql_tokenizr import tsql_tokenize


def create_dict(keys: list[str]):
    key_to_id = {}
    id_to_key = {}
    id = 1
    for key in keys:
        key_to_id[key] = id
        id_to_key[id] = key
        id += 1
    return key_to_id, id_to_key


key_dict, id_dict = create_dict([
    '<s>', '</s>',
    '<identifier>', '<number>', '<string>',
    '<<number>>',
    '<<str>>',
    '<<dict>>',
    '<<tuple>>',
    '<<arr>>',
    '[type]', '[columns]', '[froms]',
    '(', ')', '.', '+', '-', '*', '/',
    '&', '>=', '<=', '<>', '!=', '=',
    'select', 'from', 'as', 'with', 'nolock'
])


def dict_to_value_array(val, type_to_id_dict):
    if isinstance(val, str):
        return [type_to_id_dict["<<str>>"], type_to_id_dict[val]]
    if isinstance(val, list):
        arr = [type_to_id_dict["<<arr>>"], len(val)]
        for item in val:
            arr.extend(dict_to_value_array(item, type_to_id_dict))
        return arr
    if isinstance(val, tuple):
        item0, item1 = val
        arr = [type_to_id_dict["<<tuple>>"]]
        arr.extend(dict_to_value_array(item0, type_to_id_dict))
        arr.extend(dict_to_value_array(item1, type_to_id_dict))
        return arr
    if isinstance(val, dict):
        keys = val.keys()
        arr = [type_to_id_dict["<<dict>>"], len(keys)]
        for key in keys:
            value = val[key]
            arr.append(type_to_id_dict[f"[{key}]"])
            arr.extend(dict_to_value_array(value, type_to_id_dict))
        return arr
    return [type_to_id_dict["<<number>>"], val]


def value_array_to_dict(value_iter, id_to_type_dict):
    val_type = id_to_type_dict[value_iter.next()]
    if val_type == "<<str>>":
        return id_to_type_dict[value_iter.next()]
    if val_type == "<<arr>>":
        arr_len = value_iter.next()
        arr = []
        for n in range(arr_len):
            arr_item = value_array_to_dict(value_iter, id_to_type_dict)
            arr.append(arr_item)
        return arr
    if val_type == "<<tuple>>":
        item0 = value_array_to_dict(value_iter, id_to_type_dict)
        item1 = value_array_to_dict(value_iter, id_to_type_dict)
        return item0, item1
    if val_type == "<<dict>>":
        a_dict = {}
        keys_size = value_iter.next()
        for n in range(keys_size):
            key = id_to_type_dict[value_iter.next()]
            key = key.strip("[]")
            a_dict[key] = value_array_to_dict(value_iter, id_to_type_dict)
        return a_dict
    return value_iter.next()


def label_to_value(label):
    return dict_to_value_array(label, key_dict)


def label_value_to_obj(label_value):
    return value_array_to_dict(ListIter(label_value), id_dict)


def decode_label(label, sql):
    input_tokens = [token.text for token in tsql_tokenize(sql)]
    if label['type'] == 'select':
        decoded_text = ''
        decoded_text += 'SELECT '
        for idx, col in enumerate(label['columns']):
            from_index = col[0]
            input_offset = col[1]
            from_table = f"tb{from_index}.{input_tokens[input_offset]}"
            decoded_text += from_table
            if idx + 1 < len(label['columns']):
                decoded_text += ', '
        decoded_text += ' FROM '
        label_froms = label['froms']
        for idx, source in enumerate(label_froms):
            if isinstance(source, dict):
                decoded_text += '(' + decode_label(source, sql) + ')'
            else:
                table_name = input_tokens[source]
                decoded_text += f"{table_name} as tb{idx} WITH(NOLOCK)"
            if idx + 1 < len(label_froms):
                decoded_text += ', '
        return decoded_text
    return None



def query_pth_files(directory: str):
    files = os.listdir(directory)
    pth_files = [file for file in files if file.endswith('.pth')]
    pattern = r"best_model_(\d+\.\d+)"
    pth_files = [file for file in pth_files if re.match(pattern, file)]
    pth_files.sort(key=lambda file: float(re.search(pattern, file).group(1)), reverse=False)
    for file in pth_files:
        filename = os.path.join(directory, file)
        loss = float(re.search(pattern, filename).group(1))
        yield filename, loss


def keep_best_pth_files(directory: str):
    pth_files = list(query_pth_files(directory))
    for pth_file, loss in pth_files[5:]:
        os.remove(pth_file)


def load_model_pth(model):
    pth_files = list(query_pth_files("./models"))
    if len(pth_files) > 0:
        pth_file, min_loss = pth_files[0]
        model.load_state_dict(torch.load(pth_file))
        print(f"load {pth_file} file")


def read_dict_file(file: str) -> dict:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        return ast.literal_eval(content)


def is_float(word: str) -> bool:
    """Return True if string is a float number."""
    if re.match("^\\d+?\\.\\d+?$", word) is None:
        return False
    return True


def is_int(word: str) -> bool:
    """Return True if string is a int number."""
    if re.search("\\.$", word):
        return False
    if re.match("^\\d+$", word) is None:
        return False
    return True


def word_to_id(word: str) -> int:
    if is_float(word):
        return key_dict['<number>']
    if is_int(word):
        return key_dict['<number>']
    if word.startswith("'") and word.endswith("'"):
        return key_dict['<string>']
    lowered_word = word.lower()
    if lowered_word in key_dict:
        return key_dict[lowered_word]
    return key_dict['<identifier>']


def sql_to_value(sql: str):
    sql_tokens = tsql_tokenize(sql)
    sql_value = [word_to_id(token.text) for token in sql_tokens]
    sql_value = [(offset, value) for offset, value in enumerate(sql_value)]
    expanded_sql_value = [item for sublist in sql_value for item in sublist]
    return expanded_sql_value


class LabelException(Exception):
    pass


class ListIter:
    def __init__(self, a_list: list[int]):
        self.a_list = a_list
        self.index = 0

    def next(self):
        value = self.a_list[self.index]
        self.index += 1
        return value

    def eof(self):
        return self.index >= len(self.a_list)


class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.fc = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()  # 使用 ReLU 來確保輸出為非負值

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, batch_size)

        embedded = self.embedding(x)  # shape: [seq_len, batch_size, hidden_size]

        lstm_output, _ = self.lstm(embedded)  # shape: [seq_len, batch_size, hidden_size]

        # print(f"{lstm_output.shape=}")
        attention_output = self.attention(lstm_output)  # shape: [batch_size, seq_len, hidden_size * num_heads]

        # output = lstm_output[-1, :, :]  # shape: [batch_size, hidden_size]
        # print(f"{attention_output.shape=}")

        output = self.fc(attention_output)  # shape: [seq_len, batch_size, output_size]
        output = self.relu(output)
        output = torch.unsqueeze(output, 1)
        return output


class SqlTrainDataset(Dataset):
    def __init__(self, dict_file: str, max_seq_len):
        self.max_seq_len = max_seq_len
        self.data = read_dict_file(dict_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sql, label = self.data[index]
        sql_value = [key_dict['<s>']] + sql_to_value(sql) + [key_dict['</s>']]
        sql_chunks = alist_to_chunks(sql_value, max_len=self.max_seq_len)

        label_value = label_to_value(label)
        label_chunk = [key_dict['<s>']] + label_value + [key_dict['</s>']]
        label_value_chunks = alist_to_chunks(label_chunk, max_len=self.max_seq_len)
        return sql_chunks, label_value_chunks


def pad_collate_fn(batch):
    sql_chunks_list = []
    label_chunks_list = []
    for sql_chunks, label_chunks in batch:
        sql_chunks_list.append(sql_chunks)
        label_chunks_list.append(label_chunks)
    padded_sql_chunks = pad_chunks_list(sql_chunks_list)
    padded_label_chunks = pad_chunks_list(label_chunks_list)

    features = torch.as_tensor(padded_sql_chunks, dtype=torch.long)
    targets = torch.as_tensor(padded_label_chunks, dtype=torch.float32)
    return features, targets