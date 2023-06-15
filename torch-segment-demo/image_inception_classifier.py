import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

path_to_train_data_folder = './data/yolo/train'
path_to_val_data_folder = './data/yolo/train'
num_classes = 20
path_to_save_model = './models/model-inception.pth'

# 定义数据预处理的转换
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # 调整图像大小为 Inception-ResNet 的输入大小
    transforms.ToTensor(),  # 将图像转换为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像数据
])

# 加载训练数据集
train_dataset = ImageFolder(path_to_train_data_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 加载验证数据集
val_dataset = ImageFolder(path_to_val_data_folder, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32)

# 加载 Inception-ResNet 模型
model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
model.aux_logits = False  # 去除辅助分类器
model.fc = nn.Linear(2048, num_classes)  # 替换最后一层全连接层，num_classes为自定义分类的类别数

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10

for epoch in range(num_epochs):
    # 训练模式
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 验证模式
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 打印训练和验证的损失以及准确率
    print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.2f}%'.format(
        epoch + 1, num_epochs, train_loss / len(train_loader),
        val_loss / len(val_loader), 100 * correct / total))

# 保存模型
torch.save(model.state_dict(), path_to_save_model)
