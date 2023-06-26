from id_utils import generate_random_id, convert_id_to_numbers
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data import train_loader, tensor2d
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import matplotlib.pyplot as plt


# 定義模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.fc1 = nn.Linear(9, 18)
        self.fc2 = nn.Linear(18, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        self.param_mapping = {name: i for i, name in enumerate(self.state_dict().keys())}

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(x)
        x = torch.relu(self.fc3(x))
        x = torch.softmax(self.fc4(x), dim=1)
        return x

    # 預測方法
    def predict(self, id: str):
        id9 = id[0:9]
        id9_numbers = convert_id_to_numbers(id9)
        x = tensor2d([id9_numbers])
        with torch.no_grad():
            output = self.forward(x)
            _, predicted = torch.max(output.data, 1)
            return predicted.item()

    def get_gradient(self):
        # param_mapping = {name: i for i, name in enumerate(self.state_dict().keys())}
        param_mapping = self.param_mapping
        grads = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                grads.append((
                    param_mapping[name],
                    param.grad.data.norm(2).item()
                ))
        return grads


def draw_grads(grads_list, param_mapping):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for epoch, gradient in grads_list:
        for item in gradient:
            y, value = item
            ax.scatter(epoch, y, value)
    ax.set_yticks(list(param_mapping.values()))
    ax.set_yticklabels(list(param_mapping.keys()))
    return fig


def train(model, data_loader, optimizer, epochs):
    writer = SummaryWriter(log_dir='./logs')
    grads_list = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = model.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 0:  # 每2000個 mini-batches 打印一次
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        # 將梯度圖像寫入 TensorBoard
        grads = model.get_gradient()
        grads_list.append((epoch, grads))

    writer.close()
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
