import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_len, hidden_size, output_len):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_len, hidden_size, batch_first=True)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        output = lstm_output[:, -1].unsqueeze(0)
        return output

class Decoder(nn.Module):
    def __init__(self, input_len, hidden_size, output_len):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_len, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_len)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.linear(lstm_out)
        return output

def train_model(encoder, decoder, input_sequence, target_sequence):
    encoder.train()
    decoder.train()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.01)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_output = encoder(input_sequence)
    decoder_outputs = decoder(encoder_output)

    loss = loss_fn(decoder_outputs, target_sequence)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()

if __name__ == "__main__":
    seq_len = 5
    hidden = 100
    encoder = Encoder(seq_len, hidden, 1)
    decoder = Decoder(1, hidden, seq_len)

    for epoch in range(1000):
        input_sequence = torch.randint(low=0, high=400, size=(seq_len,)).float().unsqueeze(0)
        target_sequence = input_sequence
        loss = train_model(encoder, decoder, input_sequence, target_sequence)
        print(f"Epoch {epoch} loss: {loss}")

    # 使用模型來壓縮序列
    compressed_sequence = encoder(input_sequence)
    print(f"Compressed sequence: {compressed_sequence}")

    # 使用模型來解碼序列
    reconstructed_sequence = decoder(compressed_sequence)
    print(f"Reconstructed sequence: {reconstructed_sequence}")
    
    