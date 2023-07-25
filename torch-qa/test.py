import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from network import CharRNN, word_to_chunks, MultiHeadAttention, chunks_to_tensor, get_probability, WordRNN, \
    padding_row_array_list
from sql_network import keep_best_pth_files, load_model_pth, \
    ListIter, LSTMWithAttention, SqlTrainDataset, pad_collate_fn, convert_sql_txt_to_train_data, sql_to_value, infer, \
    decode_label, label_value_to_obj

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


def train(model, data_loader, criterion, num_epochs=10, device='cpu'):
    optimizer = optim.Adam(model.parameters())
    writer = SummaryWriter()
    min_loss = float('inf')

    load_model_pth(model)

    model = model.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs, targets)

            # Compute loss
            # loss = criterion(outputs, targets)
            loss = model.compute_loss(outputs, targets)

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
            torch.save(model.state_dict(), f"./models/best_model_{min_loss:.4f}.pth")
            print(f"New minimum loss {min_loss:.4f}, model saved.")
            keep_best_pth_files("./models")

    writer.close()
    print("Finished Training")



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




def test4():
    max_seq_len = 30
    # convert_sql_txt_to_train_data("./train_data/sql.txt",
    #                               max_seq_len=max_seq_len,
    #                               output_file="./train_data/sql_data.txt")

    # input_length = 20
    # input_array = torch.arange(1, input_length + 1)
    # # 将输入数据划分为长度为 50 的子序列
    # sequence_length = 10
    # num_sequences = input_length // sequence_length
    # input_sequences = input_array.view(num_sequences, sequence_length)
    # print(f"{input_sequences=}")

    dataset = SqlTrainDataset("./train_data/sql.txt", max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=pad_collate_fn)

    model = LSTMWithAttention()
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

    train(model, dataloader, loss_fn, num_epochs=100)

    sql = "select id, name, birth from p"
    sql_value = sql_to_value(sql)
    load_model_pth(model)
    #resp = infer(model, sql_value)
    print(f"{sql_value=}")
    output_seq = model.infer(sql_value)
    print(f"{output_seq=}")
    output_list = [x for x in output_seq if x != 0]
    label = label_value_to_obj(output_list)
    tgt = decode_label(label, sql)
    print(f"{tgt=}")


if __name__ == '__main__':
    #test4()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    def sequence_from_output(output_sequence):
        # 使用 softmax 函數計算每個詞的概率分佈
        probabilities = F.softmax(output_sequence, dim=2)
        # 取概率最大的詞作為預測結果
        _, sequence = torch.max(probabilities, dim=2)
        # 轉換為 Python list 形式
        sequence = sequence.squeeze().tolist()
        return sequence

    class LSTMWithAttention(nn.Module):
        def __init__(self, input_vocab_size, output_vocab_size, hidden_size, num_layers):
            super(LSTMWithAttention, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.embedding = nn.Embedding(input_vocab_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
            self.attention = nn.Linear(hidden_size, hidden_size)
            self.out = nn.Linear(hidden_size, output_vocab_size)

        def forward(self, x):
            embedded = self.embedding(x)
            lstm_output, _ = self.lstm(embedded)

            # Apply attention mechanism
            attention_scores = torch.tanh(self.attention(lstm_output))
            attention_weights = F.softmax(attention_scores, dim=1)
            context_vector = torch.bmm(attention_weights.transpose(1, 2), lstm_output)

            # Final output
            output = self.out(context_vector)
            return output


    # 使用示例
    input_vocab_size = 10000
    output_vocab_size = 10000
    hidden_size = 256
    num_layers = 2

    # 初始化模型
    model = LSTMWithAttention(input_vocab_size, output_vocab_size, hidden_size, num_layers)

    # 隨機生成一個長度為10的輸入序列 (範例中使用5維的單詞表示)
    input_sequence = torch.randint(0, input_vocab_size, (1, 10))

    # 進行推斷
    output_sequence = model(input_sequence)

    # 輸出結果
    print("輸入序列:", input_sequence)
    print("輸出序列:", output_sequence)

    output = sequence_from_output(output_sequence)
    print(output)