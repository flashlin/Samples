import ast
import os
import re

import torch
from torch import nn
from torch.utils.data import Dataset

from data_utils import create_char2index_map, create_index2char_map, overlap_split_list, create_running_list, pad_list, \
    zip_aggregate
from network import alist_to_chunks, pad_chunks_list, pad_array, pad_sequence_list
from tsql_tokenizr import tsql_tokenize
import torch.nn.functional as F


def create_map_dict(keys: list[str], start_tag, end_tag):
    key_to_id = create_char2index_map(keys, start=1)
    id_to_key = create_index2char_map(keys, start=1)
    start_id = 9998
    key_to_id[start_tag] = start_id
    key_to_id[end_tag] = start_id + 1
    id_to_key[start_id] = start_tag
    id_to_key[start_id + 1] = end_tag
    return key_to_id, id_to_key


key_dict, id_dict = create_map_dict([
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
], start_tag='<s>', end_tag='</s>')


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


def convert_sql_txt_to_train_data(raw_sql_train_file: str,
                                  max_seq_len: int,
                                  output_file: str):
    def create_chunk(prev_chunk):
        return list(prev_chunk[1:] + [0])

    data = read_dict_file(raw_sql_train_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sql, label in data:
            sql_value = [key_dict['<s>']] + sql_to_value(sql) + [key_dict['</s>']]
            label_value = [key_dict['<s>']] + label_to_value(label) + [key_dict['</s>']]
            sql_chunks = create_running_list(sql_value, max_seq_len=max_seq_len)
            label_chunks = create_running_list(label_value, max_seq_len=max_seq_len)
            for sql_chunk, label_chunk in zip_aggregate(sql_chunks, label_chunks, create_a_elem=create_chunk, create_b_elem=create_chunk):
                f.write(" ".join(map(str, sql_chunk)) + "\n")
                f.write(" ".join(map(str, label_chunk)) + "\n")


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
    def __init__(self, input_vocab_size=10000,
                 output_vocab_size=10000,
                 hidden_size=128, num_layers=3, num_heads=4, dropout=0.2):
        super(LSTMWithAttention, self).__init__()
        self.output_vocab_size = output_vocab_size
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size, dropout=dropout),
            num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size, dropout=dropout),
            num_layers
        )
        self.embedding = nn.Embedding(input_vocab_size, hidden_size, padding_idx=0)
        self.fc = nn.Linear(hidden_size, output_vocab_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, src, tgt):
        src_embed = self.embedding(src)
        encoder_output = self.encoder(src_embed)
        tgt_embed = self.embedding(tgt)
        decoder_output = self.decoder(tgt_embed, encoder_output)
        output = self.fc(decoder_output)
        return output

    def compute_loss(self, output, target_tensor):
        output_flattened = output.view(-1, self.output_vocab_size)
        target_flattened = target_tensor.view(-1)
        loss = self.criterion(output_flattened, target_flattened)
        return loss

    def infer(self, input_seq, max_seq_len, device='cpu'):
        start_index = key_dict["<s>"]
        end_index = key_dict["</s>"]
        new_input_seq = [start_index] + input_seq + [end_index]
        tgt_seq = pad_list([], max_len=max_seq_len-1) + [start_index]

        tgt_seq = torch.as_tensor(tgt_seq, dtype=torch.long).unsqueeze(0).to(device)
        output_seq = [start_index]
        running_input_seqs = create_running_list(new_input_seq, max_seq_len=max_seq_len)
        pred_end = False
        for idx, running_input_seq in enumerate(running_input_seqs):
            input_tensor = torch.as_tensor(running_input_seq, dtype=torch.long).unsqueeze(0).to(device)
            outputs = self.forward(input_tensor, tgt_seq)
            # 獲取當前時間步的預測結果
            pred_token = outputs[:, -1, :].argmax(dim=1, keepdim=True)
            print(f"{pred_token=}")
            output_seq.append(pred_token.squeeze(0).item())
            tgt_seq = torch.cat((tgt_seq, pred_token), dim=1)
            tgt_seq = tgt_seq[:, 1:]
            if pred_token.item() == end_index:
                pred_end = True
                break

        print(f"predict...")
        new_input_seq = pad_list(new_input_seq, max_len=max_seq_len)
        input_tensor = torch.as_tensor(new_input_seq, dtype=torch.long).unsqueeze(0).to(device)
        tgt_seq = torch.as_tensor(output_seq[-max_seq_len:], dtype=torch.long).unsqueeze(0).to(device)
        count = 0
        while not pred_end and count < 1024:
            input_tensor = torch.cat((input_tensor, torch.as_tensor([[0]], dtype=torch.long)), dim=1)
            input_tensor = input_tensor[:, 1:]
            print(f"{input_tensor=}")
            print(f"{output_seq=}")
            outputs = self.forward(input_tensor, tgt_seq)
            pred_token = outputs[:, -1, :].argmax(dim=1, keepdim=True)
            output_seq.append(pred_token.squeeze(0).item())
            tgt_seq = torch.cat((tgt_seq, pred_token), dim=1)
            tgt_seq = tgt_seq[:, 1:]
            if pred_token.item() == end_index:
                break
            count += 1

        return output_seq




def sequence_from_output(output_sequence):
    # 使用 softmax 函數計算每個詞的概率分佈
    probabilities = F.softmax(output_sequence, dim=2)
    # 取概率最大的詞作為預測結果
    _, sequence = torch.max(probabilities, dim=2)
    # 轉換為 Python list 形式
    sequence = sequence.squeeze().tolist()
    return sequence

class LSTMWithAttention2(nn.Module):
    def __init__(self, input_vocab_size=10000, output_vocab_size=10000, hidden_size=32, num_layers=3):
        super(LSTMWithAttention2, self).__init__()
        self.output_vocab_size = output_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_vocab_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, targets):
        embedded = self.embedding(x)
        lstm_output, _ = self.lstm(embedded)

        # Apply attention mechanism
        attention_scores = torch.tanh(self.attention(lstm_output))
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), lstm_output)

        # Final output
        output = self.out(context_vector)
        return output

    def compute_loss(self, output, target_tensor):
        output_flattened = output.view(-1, self.output_vocab_size)
        _, predicted_labels = torch.max(output_flattened, dim=1)
        target_flattened = target_tensor.view(-1)
        # return F.cross_entropy(output_flattened, target_flattened)

        # print(f"{predicted_labels.shape=}")

        loss = 0
        seq_length = target_tensor.shape[0]
        output_seq = predicted_labels[:seq_length].float().requires_grad_(True)
        target_seq = target_flattened[:seq_length].float().requires_grad_(True)
        loss += F.cross_entropy(output_seq.unsqueeze(0), target_seq.unsqueeze(0), reduction='mean')
        # loss /= target_tensor.size(0)
        return loss

        # # probabilities = F.softmax(output, dim=2)
        # # _, output = torch.max(probabilities, dim=2)
        # batch_size = target_tensor.size(0)
        # max_target_length = target_tensor.size(1)
        #
        # output_flattened = output.view(-1, self.output_vocab_size).float()
        # target_flattened = target_tensor.view(-1).float()
        #
        # print(f"{output_flattened.shape=} {target_flattened.shape=} {batch_size=}")
        #
        # # loss = self.criterion(output_flattened, target_flattened, ignore_index=0)
        # # return loss
        # mask = (target_flattened != 0).float()
        # loss = F.cross_entropy(output_flattened, target_flattened, reduction='none')
        # loss = torch.sum(loss * mask) / batch_size
        # return loss
    def infer(self, input_seq, max_seq_len):
        input_seq = [key_dict['<s>']] + input_seq
        input_tensor = torch.as_tensor([input_seq], dtype=torch.long)
        outputs = self.forward(input_tensor, None)
        predicts = sequence_from_output(outputs)
        return predicts


class SqlTrainDataset(Dataset):
    def __init__(self, dict_file: str, max_seq_len):
        self.max_seq_len = max_seq_len
        self.data = read_dict_file(dict_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sql, label = self.data[index]
        sql_value = [key_dict['<s>']] + sql_to_value(sql)
        label_value = label_to_value(label) + [key_dict['</s>']]
        return sql_value, label_value


def pad_collate_fn(batch):
    sql_seqs = []
    label_seqs = []
    max_list_len = 0
    for sql_value, label_value in batch:
        max_list_len = max(max_list_len, len(sql_value))
        sql_seqs.append(sql_value)
        max_list_len = max(max_list_len, len(label_value))
        label_seqs.append(label_value)

    padded_sql_seqs = pad_sequence_list(sql_seqs, max_list_len)
    padded_label_seqs = pad_sequence_list(label_seqs, max_list_len)

    features = torch.as_tensor(padded_sql_seqs, dtype=torch.long)
    targets = torch.as_tensor(padded_label_seqs, dtype=torch.long)
    return features, targets

