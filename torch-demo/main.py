from id_utils import generate_random_id
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data import train_loader, tensor2d


# 定義模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.softmax(self.fc5(x), dim=1)
        return x

    # 預測方法
    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x)
            _, predicted = torch.max(output.data, 1)
            return predicted


def train(net, data_loader, criterion, optimizer, epochs):
    gradient_list = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:  # 每2000個 mini-batches 打印一次
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        # 產生 3D 梯度圖
        param_mapping = {name: i for i, name in enumerate(net.state_dict().keys())}
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        gradient = []
        for name, param in net.named_parameters():
            if param.requires_grad:
                gradient.append((
                    epoch,
                    param_mapping[name],
                    param.grad.data.norm(2).item()
                ))
                # ax.scatter(epoch, param_mapping[name], param.grad.data.norm(2).item())
        gradient_list.append(gradient)                
        # plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for gradient in gradient_list:
        for item in gradient:
            epoch, y, value = item
            ax.scatter(epoch, y, value)
    ax.set_yticks(list(param_mapping.values()))
    ax.set_yticklabels(list(param_mapping.keys()))
    plt.show()

# 實例化模型
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 使用訓練數據訓練模型
# 訓練數據 'trainloader' 需要提前定義
# 創建數據集和數據加載器
#dataset = MyDataset()
#trainloader = DataLoader(dataset, batch_size=1, shuffle=True)
train(net, train_loader, criterion, optimizer, epochs=300)

id = generate_random_id()
