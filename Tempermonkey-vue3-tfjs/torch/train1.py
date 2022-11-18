import torch
from common.io import info
from lit_model import LitTranslateModel
from prepare import create_data_loader
from utils.linq_tokenizr import LINQ_VOCAB_SIZE
from utils.tsql_tokenizr import TSQL_VOCAB_SIZE
import pytorch_lightning as pl

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = create_data_loader()
    # Model
    info(f"{LINQ_VOCAB_SIZE=} {TSQL_VOCAB_SIZE=}")
    model = LitTranslateModel(LINQ_VOCAB_SIZE, TSQL_VOCAB_SIZE, device)

    # Training
    trainer = pl.Trainer(gpus=0, precision=16, limit_train_batches=1.0, max_epochs=10)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()