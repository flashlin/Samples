import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from network import CharRNN, word_to_chunks, MultiHeadAttention, chunks_to_tensor, get_probability, WordRNN, \
    word_chunks_list_to_chunks

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


"""
    inputs =[[array([129, 115, 101]), array([ 99, 116, 130)], [array([129, 105, 100)]]
    outputs=[[array([129, 115, 101]), array([ 99, 116, 130)], [array([129, 105, 100), array([0., 0., 0.)]]
"""
def padding_row_array_list(row_array_list, max_seq_len):
    max_length = 0
    for row in row_array_list:
        max_length = max(len(row), max_length)
    new_row_array_list = []
    for row in row_array_list:
        if len(row) < max_length:
            zero_array = [np.zeros(max_seq_len)] * (max_length-len(row))
            new_row_array_list.append(row + zero_array)
        else:
            new_row_array_list.append(row)
    return new_row_array_list


def test2():
    max_pad_len = 5
    model = WordRNN(output_size=3, max_pad_word_len=max_pad_len)
    raw_data = [
        ("select", 1),
        ("id", 2)
    ]

    input_chunks_list = [word_to_chunks(word, max_pad_len) for word, label in raw_data]
    print(f"{input_chunks_list=}")
    # input_chunks = word_chunks_list_to_chunks(input_chunks_list)
    # print(f"{input_chunks=}")
    input_labels_list = [label for word, label in raw_data]
    print(f"{input_labels_list=}")

    padded_inputs = padding_row_array_list(input_chunks_list, max_seq_len=max_pad_len)
    print(f"{padded_inputs=}")


    # Convert each numpy array to a tensor and collect these into a new list
    inputs = torch.tensor(padded_inputs, dtype=torch.long)
    print(f"{inputs=}")


    outputs = model(inputs)
    print(f"{outputs=}")


if __name__ == '__main__':
    test2()
