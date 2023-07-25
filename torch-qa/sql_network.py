import ast
import os
import re

import torch
from torch import nn
from torch.utils.data import Dataset

from network import alist_to_chunks, pad_chunks_list
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


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = embed_dim // num_heads
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
            x: (batch_size, seq_len, input_dim)
        """
        print(f"mulhead {x.shape=}")
        # 通过线性变换获取 query, key, value 张量
        query = self.linear_q(x)  # 形状为 (batch_size, seq_len, embed_dim)
        key = self.linear_k(x)  # 形状为 (batch_size, seq_len, embed_dim)
        value = self.linear_v(x)  # 形状为 (batch_size, seq_len, embed_dim)
        output, attn_weights = self.attention(query, key, value)
        return output


class LSTMWithAttention(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_size, output_dim,
                 num_heads, n_layers=3):
        super(LSTMWithAttention, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads  # 相當於 seq_len

        self.embedding = nn.Embedding(input_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.attention = MultiHeadAttention(embed_dim=hidden_size,
                                            num_heads=num_heads, dropout=0.2)
        self.fc = nn.Linear(hidden_size, seq_len)
        self.relu = nn.ReLU()  # 使用 ReLU 來確保輸出為非負值

    def forward(self, x):
        # [batch_size, 1, seq_len]
        print(f"input {x.shape=}")
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        # 輸入:(batch_size, seq_len)
        # 輸出:(batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        print(f"{embedded.shape=}")

        # 輸入:(batch_size, seq_len, input_size)
        # 輸出:(batch_size, seq_len, num_directions * hidden_size)
        lstm_output, _ = self.lstm(embedded)

        print(f"{lstm_output.shape=}")

        # attention_input = torch.transpose(lstm_output, 0, 1)
        attention_input = lstm_output
        print(f"{attention_input.shape=}")
        # 輸入:(seq_len, batch_size, hidden_size)
        # 輸出:(seq_len, batch_size, hidden_size)
        attention_output = self.attention(attention_input)
        # attention_output = torch.transpose(attention_output, 0, 1)
        print(f"{attention_output.shape=}")

        # output = lstm_output[-1, :, :]  # shape: [batch_size, hidden_size]
        # print(f"{attention_output.shape=}")

        fc_output = self.fc(attention_output)  # shape: [seq_len, batch_size, output_size]
        print(f"{fc_output.shape=}")
        relu_output = self.relu(fc_output)
        print(f"{relu_output.shape=}")
        output = torch.unsqueeze(relu_output, 1)
        print(f"{output.shape=}")
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
    max_list_len = 0
    for sql_chunks, label_chunks in batch:
        max_list_len = max(max_list_len, len(sql_chunks))
        sql_chunks_list.append(sql_chunks)
        max_list_len = max(max_list_len, len(label_chunks))
        label_chunks_list.append(label_chunks)

    padded_sql_chunks = pad_chunks_list(sql_chunks_list, max_list_len)
    padded_label_chunks = pad_chunks_list(label_chunks_list, max_list_len)

    features = torch.as_tensor(padded_sql_chunks, dtype=torch.long)
    targets = torch.as_tensor(padded_label_chunks, dtype=torch.float32)
    return features, targets
