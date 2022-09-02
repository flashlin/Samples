import torch
import torchvision.transforms as trns
import torch.utils.data as data
import os
import re
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image

import model

def list_files(dir_path, pattern):
    regex = re.compile(pattern)
    for name in os.listdir(dir_path):
        if regex.search(name) is None:
            continue
        fullname = os.path.join(dir_path, name)
        if os.path.isfile(fullname):
            yield fullname


def read_textfile(filename):
    with open(filename, "r") as f:
        data = f.read()
        lines = data.split("\n")
        return lines


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, transform=None):
        self.images_path = images_path
        self.transform = trns.Compose([
            trns.Resize((220, 415)),
            #trns.RandomCrop((224, 224)),
            trns.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0.1),
            #trns.GaussianBlur(11, sigma=(0.1, 2.0)),
            trns.RandomHorizontalFlip(),
            #trns.RandomVerticalFlip(0.5),
            trns.ToTensor(),
            #trns.Lambda(lambda x: x.repeat(3, 1, 1)), # 灰階轉為 RGB
            trns.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
        ])
        if transform is not None:
            self.transform = transform
        self.images = []
        for image in list_files(images_path, "\.png$"):
            self.images.append(image)
        self.labels = []
        for txt in list_files(images_path, "\.txt$"):
            lines = read_textfile(txt)
            self.labels.append(int(lines[0]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        img = Image.open(image_name).convert('RGB')
        img = self.transform(img)

        #   image = read_image(image_name)
        #   img = T.ToPILImage()(image)
        #   img = self.transform(img)
        X = img
        y = self.labels[index]
        y = torch.tensor(y, dtype=torch.long)
        return X, y



def train(dataset, model, device, loss_fn, optimizer):
    size = len(dataset)
    max_epochs = 8
    params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 1
    }
    train_loader = torch.utils.data.DataLoader(dataset, **params)
    model.train()
    for epoch in range(max_epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)         # 計算預測值
            loss = loss_fn(pred, y) # 計算損失值
            optimizer.zero_grad()   # 重設參數梯度(gradient)
            loss.backward()         # 反向傳播（backpropagation）
            optimizer.step()        # 更新參數

            # 輸出訓練過程資訊
            if epoch % 2 == 0:
                loss, current = loss.item(), epoch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    # # 將模型設定為訓練模式
    # model.train()
    # # 批次讀取資料進行訓練
    # for batch, (X, y) in enumerate(dataset):
    #     # 將資料放置於 GPU 或 CPU
    #     X, y = X.to(device), y.to(device)
        
    #     pred = model(X)         # 計算預測值
    #     loss = loss_fn(pred, y) # 計算損失值（loss）

    #     optimizer.zero_grad()   # 重設參數梯度（gradient）
    #     loss.backward()         # 反向傳播（backpropagation）
    #     optimizer.step()        # 更新參數

    #     # 輸出訓練過程資訊
    #     if batch % 100 == 0:
    #         loss, current = loss.item(), batch * len(X)
    #         print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")





def main():
    torch.manual_seed(0)

    dataset = MyDataset('D:/VDisk/Github/Samples/tf-jupyter/data')
    print(f"{len(dataset)=}")

    train_set_size = int(len(dataset) * 0.7)
    valid_set_size = int(len(dataset) * 0.2)
    test_set_size = len(dataset) - train_set_size - valid_set_size

    train_set, valid_set, test_set = data.random_split(dataset, [train_set_size, valid_set_size, test_set_size])
    print(f"{len(train_set)=}")
    print(f"{len(valid_set)=}")
    print(f"{len(test_set)=}")

    m = model.use_resnet18_numbers(1)
    train(train_set, m.model, m.device, m.loss_fn, m.optimizer)


if __name__ == '__main__':
    main()
