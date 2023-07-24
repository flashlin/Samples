import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from network import CharRNN, word_to_chunks, MultiHeadAttention, chunks_to_tensor, get_probability, WordRNN, \
    padding_row_array_list
from sql_network import dict_to_value_array, value_array_to_dict, keep_best_pth_files, load_model_pth, \
    sql_to_value, ListIter, LSTMWithAttention, SqlTrainDataset, pad_collate_fn
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


def test4():
    max_seq_len = 30
    dataset = SqlTrainDataset("./train_data/sql.txt", max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=pad_collate_fn)

    model = LSTMWithAttention(input_size=max_seq_len, output_size=200, hidden_size=200, num_heads=200)
    # load_model_pth(model)
    #outputs_data = model()
    #print(f"{outputs_data.shape=}")

    loss_fn = torch.nn.MSELoss()
    #loss = loss_fn(outputs_data, target_data)

    #rounded_outputs = outputs_data.round()
    #print(f"{target_data=}")
    #print(f"{outputs_data=}")
    #print(f"{rounded_outputs=}")

    # first_batch = outputs_data[0, :, :]
    # print(f"{first_batch.shape=}")
    # print(f"{first_batch=}")

    # rounded_tensor = first_batch.round()
    # print(f"{labels_data=}")
    # print(f"{rounded_tensor=}")

    train(model, dataloader, loss_fn)


if __name__ == '__main__':
    test4()
