import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, x):
        out, hidden = self.lstm(x)
        return hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        return out

def train_model(encoder, decoder, input_sequence, target_sequence):
    encoder.train()
    decoder.train()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.01)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = encoder(input_sequence)

    decoder_input = torch.zeros(1, 1, decoder.hidden_size)
    decoder_outputs = []
    for i in range(target_sequence.size(1)):
        decoder_output, decoder_hidden = decoder(decoder_input, encoder_hidden)
        decoder_outputs.append(decoder_output)
        decoder_input = decoder_output

    loss = loss_fn(decoder_outputs, target_sequence)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()

if __name__ == "__main__":
    input_sequence = torch.randint(0, 10, (1, 10)).float()
    target_sequence = torch.randint(0, 10, (1, 10))

    encoder = Encoder(10, 100)
    decoder = Decoder(100, 10)

    for epoch in range(10):
        loss = train_model(encoder, decoder, input_sequence, target_sequence)
        print(f"Epoch {epoch} loss: {loss}")

    # 使用模型來壓縮序列
    compressed_sequence = decoder(torch.zeros(1, 1, decoder.hidden_size), encoder(input_sequence))

    print(f"Compressed sequence: {compressed_sequence}")

    # 使用模型來解碼序列
    reconstructed_sequence = decoder(compressed_sequence, encoder(input_sequence))

    print(f"Reconstructed sequence: {reconstructed_sequence}")
    
    