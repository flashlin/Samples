import math
import os

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn as nn, optim as optim
from sklearn.metrics import accuracy_score
from common.io import info


class PositionalEncoding(nn.Module):
    """
為了使模型能夠利用序列的順序, 需要插入一些關於 tokens 在序列中相對或絕對位置的信息
因此專家提出了 “Positional Encoding” 位置編碼的概念, 使用了不同頻率的正弦和余弦函數來作為位置編碼
    Args
        d_model: Hidden dimensionality of the input.
        max_len: Maximum length of a sequence to expect.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Embeddings(nn.Module):
    """
是一個 lookup table
存儲了固定大小的 dictionary 的 word embeddings
輸入是 indices 來獲取指定 indices 的 word embedding 向量
    """
    def __init__(self, vocab: int, d_model: int = 512):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        a = self.lut(x) * math.sqrt(self.d_model)
        return a

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def flat_accuracy(preds, labels):
    # pred_flat = np.argmax(preds, axis=1).flatten()
    seq_len = preds.size(1)
    pred_seq = torch.max(preds, 1)[1]
    pred_flat = pred_seq
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

class BaseLightning(pl.LightningModule):
    def __init__(self, device=None):
        super().__init__()
        self.batch_size = 0
        self._device = device
        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr_scheduler = None
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        # self.criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def forward(self, batch):
        output = self.model(batch)
        return output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=50, max_iters=80
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._calculate_loss((y_hat, y), mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        _ = self._calculate_loss((y_hat, y), mode="val")

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        _ = self._calculate_loss((y_hat, y), mode="test")

    def _calculate_loss(self, batch, mode="train"):
        y_hat, y = batch
        loss = self.criterion(y_hat, y)
        self.log("%s_loss" % mode, loss)
        return loss

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def init_dataloader(self, dataset, batch_size):
        train_loader, val_loader = dataset.create_dataloader(batch_size)
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = val_loader

def start_train(model_type, device=None,
                checkpoint_path="./output",
                train_task_name="TrainTask",
                max_epochs=10,
                **kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir = os.path.join(checkpoint_path, train_task_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        #callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="train_loss")],
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=max_epochs,
        gradient_clip_val=5,
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(checkpoint_path, f"{train_task_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        info("Found pretrained model, loading...")
        model = model_type.load_from_checkpoint(pretrained_filename)
    else:
        model = model_type(**kwargs)
        train_loader = model.train_dataloader()
        val_loader = model.val_dataloader()
        # trainer.fit(model, train_loader, val_loader)
        # for batch, idx in train_loader:
        #     info(f" {batch=}")
        trainer.fit(model)

    # Test best model on validation and test set
    test_loader = model.test_dataloader()
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    # result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}
    model = model.to(device)
    return model #, result
