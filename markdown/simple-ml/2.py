import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(()))
        self.bias = nn.Parameter(torch.randn(()))

    def forward(self, x):
        x = torch.clamp(x * self.weight, -1, 1)
        return torch.asin(x) + self.bias
    
model_file = './models/best.pth'
device = 'cpu' 
def train(model, criterion, optimizer, data_loader, num_epochs):
    model.train()  # Set the model to training mode
    best_loss = float('inf')
    for epoch in range(num_epochs):
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)  # Forward pass
            loss = criterion(outputs, y)  # Compute loss

            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), model_file)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


class MyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.from_numpy(x_data).float()
        self.y_data = torch.from_numpy(y_data).float()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


def linspace(start, end, num):
    return np.linspace(start, end, num)

# Use the linspace function to generate y values
ys = linspace(0, 4*np.pi, 2000)  # y range from 0 to 4π
# Generate x values as the sine of y values
x_data = np.sin(ys)
y_data = ys


my_dataset = MyDataset(x_data, y_data)

model = MyModel()
if os.path.isfile(model_file):
    model.load_state_dict(torch.load(model_file))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
data_loader = DataLoader(my_dataset, batch_size=32, shuffle=True)
num_epochs = 500
train(model, criterion, optimizer, data_loader, num_epochs)

y = model(100)
print(y)
    