import torch
from torch.utils.data import random_split, DataLoader
from common.io import info
from lit_model import LitTranslateModel
from prepare import Linq2TSqlDataset
from utils.linq_tokenizr import LINQ_VOCAB_SIZE
from utils.tsql_tokenizr import TSQL_VOCAB_SIZE
import pytorch_lightning as pl

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    # dataset = MNIST("", train=True, download=True, transform=transforms.ToTensor())
    csv_file_path = "./output/linq-sample.csv"
    dataset = Linq2TSqlDataset(csv_file_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=32)
    val_loader = DataLoader(val_data, batch_size=32)

    # Model
    info(f"{LINQ_VOCAB_SIZE=} {TSQL_VOCAB_SIZE=}")
    model = LitTranslateModel(LINQ_VOCAB_SIZE, TSQL_VOCAB_SIZE, device)

    # Training
    trainer = pl.Trainer(gpus=0, precision=16, limit_train_batches=1.0, max_epochs=10)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()