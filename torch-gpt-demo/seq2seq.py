import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

def encode_text_to_numbers(text):
    numbers = []
    for char in text:
        numbers.append(ord(char))
    return numbers

class CustomerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input, output = self.data[index]
        return encode_text_to_numbers(input), encode_text_to_numbers(output)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
class SQLTransformer(nn.Module):
    def __init__(self):
        super(SQLTransformer, self).__init__()
        self.embedding = nn.Embedding(256, 128)  # ASCII 值為 255，使用 128 維的嵌入向量
        self.transformer = nn.Transformer(d_model=128, num_encoder_layers=4, num_decoder_layers=4)
        self.linear = nn.Linear(128, 256)

    def forward(self, x):
        embedded = self.embedding(x)
        encoded = self.transformer(embedded, embedded)
        output = self.linear(encoded)
        return output    

def loss_fn(outputs, labels):
    # outputs = torch.reshape(outputs, (-1,))
    # outputs = torch.unsqueeze(outputs, 0)
    # labels = torch.squeeze(labels)
    # labels = torch.unsqueeze(labels, 0)
    print(f'{outputs.shape=}')
    print(f'{labels.shape=}')
    return torch.nn.CrossEntropyLoss(reduction='mean')(outputs, labels)

def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, dataloader, loss_fn, optimizer, epochs):
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs = torch.tensor(inputs, dtype=torch.long)
            outputs = model(inputs)
            labels = torch.tensor(labels, dtype=torch.long)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def infer(model, input):
    outputs = model(input)
    return outputs.item()

# Load the data
data = [
    ("select name from customer", "SELECT name FROM customer WITH(NOLOCK)"),
    ("SELECT ID from customer", "SELECT ID FROM customer WITH(NOLOCK)"),
    ("select id, Name from customer", "SELECT id, Name FROM customer WITH(NOLOCK)")
]

dataset = CustomerDataset(data)
dataloader = DataLoader(dataset, batch_size=1)

model = SQLTransformer()
train(model, dataloader, loss_fn, optimizer, 10)
save_model(model, "model.pt")

infer_fn = lambda input: infer(model, input)
print(infer_fn("select Addr from home"))