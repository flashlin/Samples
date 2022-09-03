import torch
import torchvision.transforms as trns
import torch.utils.data as data
import os
import re
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image
import sys

import model


def query_files(dir_path: str, pattern: str):
    # print(f"query_files {dir_path=} {pattern=}")
    regex = re.compile(pattern)
    for name in os.listdir(dir_path):
        if regex.search(name) is None:
            continue
        fullname = os.path.join(dir_path, name)
        if os.path.isfile(fullname):
            yield fullname


def query_files_by_paths(pattern: str, dir_paths):
    for dir_path in dir_paths:
        for file in query_files(dir_path, pattern):
            yield file


def list_files_by_paths(pattern: str, dir_paths):
    files = []
    for file in query_files_by_paths(pattern, dir_paths):
        files.append(file)
    return files


def concat_list(*list_files_list):
    for file_list in list_files_list:
        for file in file_list:
            yield file


def read_textfile(filename):
    with open(filename, "r") as f:
        data = f.read()
        lines = data.split("\n")
        return lines


def query_epoch_checkpoints(path: str, epoch: int):
    loss_pattern = r'\d+.\d+'
    regex = re.compile(loss_pattern)
    epoch_file_pattern = rf"{epoch+1:05d}_({loss_pattern})_model\.pt$"
    for pt_file in query_files(path, epoch_file_pattern):
        print(f"query {pt_file=}")
        loss = float(regex.search(pt_file).group(1))
        yield pt_file, loss


def list_epoch_checkpoints(path: str, epoch: int):
    files = []
    for file in query_epoch_checkpoints(path, epoch):
        files.append(file)
    return files


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images_paths):
        self.transform = trns.Compose([
            trns.Resize((220, 415)),
            # trns.RandomCrop((224, 224)),
            # trns.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0.1),
            # trns.GaussianBlur(11, sigma=(0.1, 2.0)),
            # trns.RandomHorizontalFlip(),
            # trns.RandomVerticalFlip(0.5),
            trns.ToTensor(),
            # trns.Lambda(lambda x: x.repeat(3, 1, 1)), # 灰階轉為 RGB
            trns.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
        ])
        self.images = []
        for image in list_files_by_paths("\.png$", images_paths):
            self.images.append(image)
        self.labels = []
        for txt in list_files_by_paths("\.txt$", images_paths):
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
        y = torch.tensor(y, dtype=torch.float32)
        return X, y


def train(epoch, dataset, model, device, loss_fn, optimizer):
    size = len(dataset)
    params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 1
    }
    train_loader = torch.utils.data.DataLoader(dataset, **params)
    model.train()
    train_loss_tmp = 0
    train_loss_avg = 0
    # for X, y in train_loader:
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)  # 計算預測值
        # print(f"{X.shape=} {y.shape=} {pred.shape=}")
        loss = loss_fn(pred, y)  # 計算損失值
        optimizer.zero_grad()  # 重設參數梯度(gradient)
        loss.backward()  # 反向傳播（backpropagation）
        optimizer.step()  # 更新參數

        # 輸出訓練過程資訊
        loss, current = loss.item(), epoch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        train_loss_tmp += loss
        train_loss_avg = train_loss_tmp / (batch_idx + 1)
    print(f"{epoch=} loss: {train_loss_avg:>7f}")
    # save_checkpoint(epoch, model, train_loss_avg, optimizer, f"./checkpoints/{epoch+1:05d}_{train_loss_avg:1.5f}_model.pt")
    save_best_checkpoint(epoch, model, train_loss_avg, optimizer,
                         f"./checkpoints/{epoch + 1:05d}_{train_loss_avg:1.5f}_model.pt")


def save_checkpoint(epoch, model, loss, optimizer, path="model.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def save_best_checkpoint(epoch: int, model, loss, optimizer, path="model.pt"):
    print(" ============================ ")
    pt_files = list_epoch_checkpoints("./data", epoch)
    for (pt_file, old_loss) in pt_files:
        print(f"{pt_file=} {old_loss=}")
        if old_loss > loss:
            os.remove(pt_file)
            continue
        return
    save_checkpoint(epoch, model, loss, optimizer, path)


def get_best_checkpoint_file(path):
    pt_files = query_files(path, "\\.pt$")
    regex = re.compile(r'\d+_(\d+\.\d+)_model\.pt$')
    min_loss = sys.maxsize
    file = ""
    for pt_file in pt_files:
        match = regex.search(pt_file)
        if match is None:
            continue
        loss = float(match.group(1))
        if loss < min_loss:
            min_loss = loss
            file = pt_file
    return file


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train_loop(dataset, model, device, loss_fn, optimizer):
    max_epochs = 10
    model.train()
    for epoch in range(max_epochs):
        train(epoch, dataset, model, device, loss_fn, optimizer)


def main():
    torch.manual_seed(0)

    dataset = MyDataset([
        # 'D:/VDisk/Github/Samples/tf-jupyter/data',
        'D:/VDisk/Github/Samples/pytorch_sample1/data',
        'D:/VDisk/Github/Samples/pytorch_sample1/data/data1',
    ])
    print(f"{len(dataset)=}")

    train_set_size = int(len(dataset) * 0.7)
    valid_set_size = int(len(dataset) * 0.2)
    test_set_size = len(dataset) - train_set_size - valid_set_size

    train_set, valid_set, test_set = data.random_split(dataset, [train_set_size, valid_set_size, test_set_size])
    print(f"{len(train_set)=}")
    print(f"{len(valid_set)=}")
    print(f"{len(test_set)=}")

    m = model.use_resnet18_numbers(1)

    best_pt_file = get_best_checkpoint_file("./checkpoints")
    load_checkpoint(m.model, m.optimizer, best_pt_file)

    train_loop(train_set, m.model, m.device, m.loss_fn, m.optimizer)


if __name__ == '__main__':
    main()
