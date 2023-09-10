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
    info(f"{LINQ_VOCAB_SIZE=} {TSQL_VOCAB_SIZE=}")
    encoder = LitEncoder()

if __name__ == "__main__":
    main()