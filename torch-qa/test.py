import re

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from network import CharRNN, word_to_chunks, MultiHeadAttention, chunks_to_tensor, get_probability, WordRNN, \
    padding_row_array_list
from tsql_tokenizr import tsql_tokenize

MAX_WORD_LEN = 5


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


def train(model, data_loader, num_epochs=10):
    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Initialize a SummaryWriter for TensorBoard
    writer = SummaryWriter()

    # Track the minimum loss
    min_loss = float('inf')

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Iterate over epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0

        # Iterate over batches
        for i, (inputs, targets) in enumerate(data_loader):
            # Move data to GPU if available
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
            torch.save(model.state_dict(), f"best_model_{min_loss:.2f}.pth")
            print(f"New minimum loss {min_loss:.2f}, model saved.")

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
    '<identifier>',
    '<number>',
    '<string>',
    '(', ')', '.', '+', '-', '*', '/',
    '&', '>=', '<=', '<>', '!=', '=',
    'select', 'from', 'as', 'with', 'nolock'
]
word_to_id_dict = {}
id = 1
for key in sql_keywords:
    word_to_id_dict[key] = id + 1
    id += 1


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


def test2():
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
                columns: [3.1],             //預測幾個欄位, [哪一個位置.哪一個位置] as 哪一個位置
                from: [3]                   //預測幾個table,
                                                [position, table start, table end] or
                                                [select...]
            },
            {
                type: select,
                columns: [3.1],
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

    sql_value1 = sql_to_value("select id from customer")
    print(f"{sql_value1=}")


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


if __name__ == '__main__':
    test2()
