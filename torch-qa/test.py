import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from data_utils import create_running_list
from network import CharRNN, word_to_chunks, MultiHeadAttention, chunks_to_tensor, get_probability, WordRNN, \
    padding_row_array_list
from sql_network import keep_best_pth_files, load_model_pth, \
    ListIter, LSTMWithAttention, SqlTrainDataset, pad_collate_fn, convert_sql_txt_to_train_data, sql_to_value, \
    decode_label, label_value_to_obj, LSTMWithAttention2

MAX_WORD_LEN = 5

def train(model, data_loader, num_epochs=10, device='cpu'):
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




def test4():
    max_seq_len = 30

    convert_sql_txt_to_train_data("./train_data/sql.txt",
                                  max_seq_len=max_seq_len,
                                  output_file="./train_data/sql_data.txt")

    # input_length = 20
    # input_array = torch.arange(1, input_length + 1)
    # # 将输入数据划分为长度为 50 的子序列
    # sequence_length = 10
    # num_sequences = input_length // sequence_length
    # input_sequences = input_array.view(num_sequences, sequence_length)
    # print(f"{input_sequences=}")

    dataset = SqlTrainDataset("./train_data/sql.txt", max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=pad_collate_fn)

    model = LSTMWithAttention2()
    train(model, dataloader, num_epochs=100, device='cuda')

    sql = "select id, name, birth from p"
    sql_value = sql_to_value(sql)
    load_model_pth(model)
    model.to('cpu')
    print(f"{sql_value=}")
    output_seq = model.infer(sql_value, max_seq_len)
    print(f"{output_seq=}")
    output_list = [x for x in output_seq if x != 0]
    label = label_value_to_obj(output_list)
    tgt = decode_label(label, sql)
    print(f"{tgt=}")


if __name__ == '__main__':
    test4()
