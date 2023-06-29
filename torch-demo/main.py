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


class SigmodNeuron(nn.Module):
    def __init__(self, dim):
        super(SigmodNeuron, self).__init__()
        self.linear = nn.Linear(dim, 1, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        output = self.linear(x)
        output = self.activation(output)
        output = output * 10
        output = torch.floor(output)
        return output
    
    
class IntegerWeightInit:
    def __call__(self, tensor):
        return nn.init.uniform_(tensor, -1, 1).round()

class IntegerLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(IntegerLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init = IntegerWeightInit()

    def forward(self, x):
        weight = self.linear.weight
        weight = self.init(weight)
        output = self.linear(x)
        return output    


class RestrictedNeuron(nn.Module):
    def __init__(self):
        super(RestrictedNeuron, self).__init__()
        self.linear = IntegerLinear(9, 1, bias=False)

    def forward(self, x):
        second_element = x[:, 1]  # 提取第二個元素
        mask = ((second_element != 1) & (second_element != 2)).unsqueeze(1).float()  # 創建遮罩
        output = self.linear(x)
        output = output + mask * 1e9  # 將損失增加到非常大的值
        return output


def create_mlp_layers(input_dim, kernels, output_dim):
    layers = []
    prev_dim = input_dim
    for i, dim in enumerate(kernels):
        layers.append(nn.Linear(prev_dim, dim))
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)

def execute_mlp_layers(layers, x):
    for i, layer in enumerate(layers):
        x = layer(x)
        if i < len(layers) - 1:
            x = torch.relu(x)
    return x

# 定義模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()
        self.fc1 = IntegerLinear(9, 1, bias=False)
        self.sigmod = SigmodNeuron(1)
        self.param_mapping = {name: i for i, name in enumerate(self.state_dict().keys())}

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmod(x)
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
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train(model, train_loader, optimizer, epochs=300)

    id = generate_random_id()
    output = model.predict(id)
    print(f'{id} => {output}')
