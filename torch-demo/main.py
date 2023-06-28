import os

from id_utils import generate_random_id, convert_id_to_numbers
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data import train_loader, tensor2d
import matplotlib.pyplot as plt


class CnnLstmMlpNet(nn.Module):
    def __init__(self):
        super(CnnLstmMlpNet, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveMaxPool1d(output_size=1),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, 10)
        self.param_mapping = {name: i for i, name in enumerate(self.state_dict().keys())}

    def forward(self, x):
        x = x.unsqueeze(1)  # 將輸入形狀轉換為[N, 1, 9]
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # 將通道維度移到最後一維，以符合LSTM的輸入形狀要求
        _, (x, _) = self.lstm(x)
        x = x.squeeze(0)
        x = self.fc(x)
        return x

    # 預測方法
    def predict(self, id: str):
        id9 = id[0:9]
        id9_numbers = convert_id_to_numbers(id9)
        x = tensor2d([id9_numbers])

        self.eval()  # 將模型設置為評估模式
        with torch.no_grad():
            output = self.forward(x)
            _, predicted = torch.max(output, 1)
        return predicted

    def get_gradient(self):
        # param_mapping = {name: i for i, name in enumerate(self.state_dict().keys())}
        param_mapping = self.param_mapping
        grads = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                grads.append((
                    name,
                    param_mapping[name],
                    param.grad.data.norm(2).item()
                ))
        return grads



# 定義模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()
        self.fc1 = nn.Linear(9, 10)
        self.fc2 = nn.Linear(10, 3)
        self.fc3 = nn.Linear(3, 1)
        self.param_mapping = {name: i for i, name in enumerate(self.state_dict().keys())}

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    # 預測方法
    def predict(self, id: str):
        id9 = id[0:9]
        id9_numbers = convert_id_to_numbers(id9)
        x = tensor2d([id9_numbers])
        with torch.no_grad():
            output = self.forward(x)
            # _, predicted = torch.max(output.data, 1)
            #return predicted.item()
            return output

    def get_gradient(self):
        # param_mapping = {name: i for i, name in enumerate(self.state_dict().keys())}
        param_mapping = self.param_mapping
        grads = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                grads.append((
                    name,
                    param_mapping[name],
                    param.grad.data.norm(2).item()
                ))
        return grads


def draw_grads(grads_list, param_mapping):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for epoch, gradient in grads_list:
        for item in gradient:
            _, y, value = item
            ax.scatter(epoch, y, value)
    ax.set_yticks(list(param_mapping.values()))
    ax.set_yticklabels(list(param_mapping.keys()))
    return fig


def train(model, data_loader, optimizer, epochs):
    # writer = SummaryWriter(log_dir='./logs')
    grads_list = []

    model_file = 'models/best_model.pth'
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))

    best_loss = float('inf')
    best_state_dict = model.state_dict()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            labels = labels.view(-1, 1).float()
            loss = model.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_file)
        print('[%d] loss: %.5f %.5f' % (epoch + 1, running_loss, best_loss))

        # 將梯度圖像寫入 TensorBoard
        grads = model.get_gradient()
        grads_list.append((epoch, grads))

    # writer.close()
    draw_grads(grads_list, model.param_mapping)
    plt.show()


if __name__ == '__main__':
    # 實例化模型
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.4)

    train(model, train_loader, optimizer, epochs=100)

    id = generate_random_id()
    output = model.predict(id)
    print(f'{id} => {output}')
