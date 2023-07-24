import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from network import CharRNN, word_to_chunks, MultiHeadAttention, chunks_to_tensor, get_probability, WordRNN, \
    padding_row_array_list, alist_to_chunks, padding_alist_chunks_list
from tsql_tokenizr import tsql_tokenize

MAX_WORD_LEN = 5


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
        arr = [type_to_id_dict["<<dict>>"]]
        for key in val.keys():
            value = val[key]
            arr.append(type_to_id_dict[f"[{key}]"])
            arr.extend(dict_to_value_array(value, type_to_id_dict))
        return arr
    return [type_to_id_dict["<<number>>"], val]


def value_array_to_dict(value_iter, id_to_type_dict):

    def get_value():
        val_type = id_to_type_dict[value_iter.next()]
        if val_type == "<<str>>":
            return id_to_type_dict[value_iter.next()]
        if val_type == "<<arr>>":
            arr_len = value_iter.next()
            arr = []
            for n in range(arr_len):
                arr_item = get_value()
                arr.append(arr_item)
            return arr
        if val_type == "<<tuple>>":
            item0 = get_value()
            item1 = get_value()
            return item0, item1
        if val_type == "<<dict>>":
            return value_array_to_dict(value_iter, id_to_type_dict)
        return value_iter.next()

    b_dict = {}
    while not value_iter.eof():
        key = id_to_type_dict[value_iter.next()]
        key = key.strip("[]")
        b_dict[key] = get_value()

    return b_dict


def test(text):
    s2 = word_to_chunks(text, max_len=MAX_WORD_LEN)
    print(f"{s2}")

    print("")
    model = CharRNN(256 + 2, 3, 32)
    s2 = word_to_chunks("select")
    input = chunks_to_tensor(s2)
    print(f"{input=}")
    outputs = model(input)
    print(f"output {outputs=}")

    probabilities = F.softmax(outputs, dim=1)
    print(f"{probabilities=}")
    predicted_classes = torch.argmax(outputs, dim=1)
    print(f"{predicted_classes=}")

    # 採用平均
    v_mean = torch.mean(outputs, dim=0)
    predicted_class = torch.argmax(v_mean)
    print(f"平均 {predicted_class=}")

    predicted_class = torch.argmax(outputs, dim=1)
    print(f"直接 {predicted_class=}")

    weights = torch.nn.functional.softmax(torch.rand(5), dim=0)  # 假设权重是随机的
    final_output = torch.sum(outputs * weights.unsqueeze(-1), dim=0)
    print(f"{final_output}")

    m1 = MultiHeadAttention(3, 3)
    outputs = outputs.unsqueeze(1)  # 這個例子使用的是一個單一的樣本（批量大小為1）。在實際使用時，你可能會一次處理多個樣本，那麼批量大小就大於1了
    outputs2 = m1(outputs)
    print(f"m {outputs2=}")

    n = get_probability(outputs2)
    print(f"{n=}")


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


def train(model, data_loader, criterion, num_epochs=10):
    optimizer = optim.Adam(model.parameters())
    writer = SummaryWriter()
    min_loss = float('inf')

    load_model_pth(model)

    if torch.cuda.is_available():
        model = model.cuda()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(data_loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Record the average loss of this epoch to TensorBoard
        avg_loss = running_loss / len(data_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)

        print(f"Loss: {avg_loss}")

        # Save the model weights if this epoch gives a new minimum loss
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model.state_dict(), f"./models/best_model_{min_loss:.2f}.pth")
            print(f"New minimum loss {min_loss:.2f}, model saved.")
            keep_best_pth_files("./models")

    writer.close()
    print("Finished Training")


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


def convert_to_float(s):
    if is_float(s):
        return float(s)
    return None


sql_keywords = [
    '<s>', '</s>',
    '<identifier>',
    '<number>',
    '<string>',
    '(', ')', '.', '+', '-', '*', '/',
    '&', '>=', '<=', '<>', '!=', '=',
    'select', 'from', 'as', 'with', 'nolock'
]
word_to_id_dict, _ = create_dict(sql_keywords)


def word_to_id(word: str) -> int:
    if is_float(word):
        return word_to_id_dict['<number>']
    if is_int(word):
        return word_to_id_dict['<number>']
    if word.startswith("'") and word.endswith("'"):
        return word_to_id_dict['<string>']
    lowered_word = word.lower()
    if lowered_word in word_to_id_dict:
        return word_to_id_dict[lowered_word]
    return word_to_id_dict['<identifier>']


def text_to_np(text: str):
    pass


def sql_to_value(sql: str):
    sql_tokens = tsql_tokenize(sql)
    sql_value = [word_to_id(token.text) for token in sql_tokens]
    sql_value = [(offset, value) for offset, value in enumerate(sql_value)]
    expanded_sql_value = [item for sublist in sql_value for item in sublist]
    return expanded_sql_value


label_type_dict, label_type_id_dict = create_dict([
    '<s>', '</s>', 'offset', 'select'
])


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


def label_columns_fn():
    def to_value(label_columns):
        values = []
        columns_size = len(label_columns)
        values.append(columns_size)
        for from_index, input_offset in label_columns:
            values.append(from_index)
            values.append(input_offset)
        return values

    def from_value(label_columns_value: ListIter):
        result = []
        columns_size = label_columns_value.next()
        for n in range(columns_size):
            from_index = label_columns_value.next()
            input_offset = label_columns_value.next()
            result.append([from_index, input_offset])
        return result

    return to_value, from_value


def label_froms_fn():
    def to_value(label_froms):
        values = []
        from_size = len(label_froms)
        values.append(from_size)
        for data in label_froms:
            from_type_name = data[0]
            values.append(label_type_dict[from_type_name])
            if from_type_name == 'offset':
                values.append(data[1])
            elif from_type_name == 'select':  # select type
                values.append(label_to_value(data[1]))
            else:
                raise LabelException(f"not support {from_type_name=}")
        return values

    def from_value(values: ListIter):
        result = []
        from_size = values.next()
        for n in range(from_size):
            one_from = []
            from_type = values.next()
            from_type_name = label_type_id_dict[from_type]
            one_from.append(from_type_name)
            if from_type == label_type_dict['offset']:
                one_from.append(values.next())
                result.append(one_from)
            elif from_type == label_type_dict['select']:
                select_obj = from_value(values)
                one_from.append(select_obj)
                result.append(one_from)
            else:
                raise LabelException(f"deserialize not support {from_type=}")
        return result

    return to_value, from_value


def label_to_value(label):
    values = []
    label_type_name = label['type']
    label_type = label_type_dict[label_type_name]
    values.append(label_type)
    if label_type_name == 'select':
        label_columns_to, _ = label_columns_fn()
        label_columns_value = label_columns_to(label['columns'])
        values.extend(label_columns_value)
        label_froms_to, _ = label_froms_fn()
        label_froms_value = label_froms_to(label['froms'])
        values.extend(label_froms_value)
    return values


def label_value_to_obj(label_value):
    label_value = ListIter(label_value)
    label_type = label_value.next()
    if label_type == label_type_dict['select']:
        label = {
            'type': 'select',
            'columns': [],
            'froms': []
        }
        _, label_columns_from = label_columns_fn()
        label['columns'] = label_columns_from(label_value)
        _, label_froms_from = label_froms_fn()
        label['froms'] = label_froms_from(label_value)
        return label
    return None


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
        for idx, source in enumerate(label['froms']):
            source = ListIter(source)
            from_type = source.next()
            if from_type == 'offset':
                table_name = input_tokens[source.next()]
                decoded_text += f"{table_name} as tb{idx} WITH(NOLOCK)"
            elif from_type == 'select':
                decoded_text += '('
                decoded_text += decode_label(source.next(), input_tokens)
                decoded_text += ')'
            else:
                raise Exception(f"not support {from_type}")
            if idx + 1 < len(label['froms']):
                decoded_text += ', '
        return decoded_text
    return None


def test2():
    max_pad_len = 5
    model = WordRNN(output_size=3, max_pad_word_len=max_pad_len)
    raw_data = [
        ("select", 1),
        ("id", 2)
    ]

    input_chunks_list = [word_to_chunks(word, max_pad_len) for word, label in raw_data]
    padded_inputs = padding_row_array_list(input_chunks_list, max_seq_len=max_pad_len)
    inputs = torch.tensor(padded_inputs, dtype=torch.long)
    print(f"{inputs=}")

    input_labels_list = [label for word, label in raw_data]

    outputs = model(inputs)
    print(f"{outputs=}")


def test3():
    """
        inputs:
        [
            "select id from customer",
            "select id from ( select id from extra_customer )",
        ]

        [
            [0, select, 1, identifier, 2, from, 3, identifier],
            [0, select, 1, identifier, 2, from, 3, (, 4, select, 5, identifier, 6, from, 7, identifier, 8, )],
        ]

        [
            {
                type: select,               //[select/position]
                columns: [0.1],             //預測幾個欄位, [哪一個位置.哪一個位置] as 哪一個位置
                from: [(0, 3)]              //[position, table start, table end] or
                                                [select...]
            },
            {
                type: select,
                columns: [0.1],
                from: [
                    {
                        type: select,
                        columns: [7.5],
                        from: [7]
                    }
                ]
            }
        ]
    """

    sql = "select id from customer"
    sql_value = sql_to_value(sql)
    print(f"{sql_value=}")

    label = {
        'type': 'select',
        'columns': [[0, 1]],
        'froms': [['offset', 3]]
    }
    print(f"{label=}")
    label_value = label_to_value(label)
    print(f"{label_value=}")
    label_obj = label_value_to_obj(label_value)
    print(f"{label_obj=}")

    label_text = decode_label(label_obj, sql)
    print(f"{label_text=}")


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


def test4():
    raw_data = [
        ("select id from cust", {
            'type': 'select',
            'columns': [(0, 1)],
            'froms': [3]
        }),
        ("select id , name from cust", {
            'type': 'select',
            'columns': [(0, 1), (0, 3)],
            'froms': [5]
        }),
        ("select id , name from ( select id, name from cust )", {
            'type': 'select',
            'columns': [(0, 1), (0, 3)],
            'froms': [{
                'type': 'select',
                'columns': [(0, 7), (0, 9)],
                'froms': [11]
            }]
        }),
    ]

    for sql, label in raw_data:
        value = dict_to_value_array(label, key_dict)
        print(f"{label=}")
        print(f"{value=}")
        obj = value_array_to_dict(ListIter(value), id_dict)
        print(f"{obj=}")
        print("")
    return


    max_seq_len = 30
    features_data = []
    labels_data = []
    for sql, label in raw_data:
        sql_value = [word_to_id_dict['<s>']] + sql_to_value(sql) + [word_to_id_dict['</s>']]
        sql_chunks = alist_to_chunks(sql_value, max_len=max_seq_len)
        feature = [sql_chunks]
        features_data.append(feature)

        label_value = label_to_value(label)
        label_obj = label_value_to_obj(label_value)
        label_text = decode_label(label_obj, sql)
        # print(f"{label_text=}")
        label_chunk = [label_type_dict['<s>']] + label_value + [label_type_dict['</s>']]
        label_value_chunks = alist_to_chunks(label_chunk, max_len=max_seq_len)
        target = [label_value_chunks]
        labels_data.append(target)

    padded_features_data = padding_alist_chunks_list(features_data)
    padded_labels_data = padding_alist_chunks_list(labels_data)
    # print(f"{padded_features_data=}")

    # input_data = torch.LongTensor(padded_features_data)
    input_data = torch.as_tensor(padded_features_data, dtype=torch.long)
    # target_data = torch.LongTensor(padded_labels_data)
    target_data = torch.as_tensor(padded_labels_data, dtype=torch.float32)
    data_loader = []
    data_loader.append((input_data, target_data))
    print(f"{target_data.shape=}")

    model = LSTMWithAttention(input_size=max_seq_len, output_size=200, hidden_size=200, num_heads=200)
    load_model_pth(model)
    outputs_data = model(input_data)
    print(f"{outputs_data.shape=}")

    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(outputs_data, target_data)
    print(f"{loss=}")

    rounded_outputs = outputs_data.round()
    print(f"{target_data=}")
    print(f"{outputs_data=}")
    print(f"{rounded_outputs=}")

    # first_batch = outputs_data[0, :, :]
    # print(f"{first_batch.shape=}")
    # print(f"{first_batch=}")

    # rounded_tensor = first_batch.round()
    # print(f"{labels_data=}")
    # print(f"{rounded_tensor=}")

    # train(model, data_loader, loss_fn)


if __name__ == '__main__':
    test4()
