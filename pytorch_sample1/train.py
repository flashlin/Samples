import torch
import torchvision.transforms as trns
import torch.utils.data as data
import os
import re
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image

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
            trns.GaussianBlur(11, sigma=(0.1, 2.0)),
            trns.RandomHorizontalFlip(),
            #trns.RandomVerticalFlip(0.5),
            trns.ToTensor(),
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
        return X, y


def main():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Parameters
    params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 1
    }
    max_epochs = 10

    dataset = MyDataset('D:/VDisk/Github/Samples/tf-jupyter/data')
    print(f"{len(dataset)=}")

    train_set_size = int(len(dataset) * 0.7)
    valid_set_size = int(len(dataset) * 0.2)
    test_set_size = len(dataset) - train_set_size - valid_set_size

    train_set, valid_set, test_set = data.random_split(dataset, [train_set_size, valid_set_size, test_set_size])
    print(f"{len(train_set)=}")
    print(f"{len(valid_set)=}")
    print(f"{len(test_set)=}")

    # Generators
    training_generator = torch.utils.data.DataLoader(train_set, **params)

    # Loop over epochs
    # for epoch in range(max_epochs):
    #     # Training
    #     for local_batch, local_labels in training_generator:
    #         # Transfer to GPU
    #         local_batch, local_labels = local_batch.to(
    #             device), local_labels.to(device)
    #         print(f"{local_batch} {local_labels}")
    #         # Model computations

    #    #  # Validation
    #    #  with torch.set_grad_enabled(False):
    #    #      for local_batch, local_labels in validation_generator:
    #    #          # Transfer to GPU
    #    #          local_batch, local_labels = local_batch.to(device), local_labels.to(device)

    #    #          # Model computations


if __name__ == '__main__':
    main()
