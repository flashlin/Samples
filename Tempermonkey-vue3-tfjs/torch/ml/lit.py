import inspect
import math
import os
import re
import shutil

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn as nn, optim as optim
from sklearn.metrics import accuracy_score
from common.io import info, get_directory_list_by_pattern, get_file_list_by_pattern, info_error, is_file_exists
from utils.tsql_tokenizr import TSQL_VOCAB_SIZE
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    """
為了使模型能夠利用序列的順序, 需要插入一些關於 tokens 在序列中相對或絕對位置的信息
因此專家提出了 “Positional Encoding” 位置編碼的概念, 使用了不同頻率的正弦和余弦函數來作為位置編碼
    Args
        d_model: Hidden dimensionality of the input.
        max_len: Maximum length of a sequence to expect.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
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


class PositionalEncoding2(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        # pe[:, 1::2] = torch.cos(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一個參數不參與梯度下降，但又希望保存 model 的时候將其保存下来
        # 這個時候就可以用 register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 為 embedding 後的 inputs，例如(1, 7, 128)，batch size:1, 單字: 7, 單詞維度 128
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
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


def scaled_dot_product(q, k, v, mask=None):
    """
    所謂的注意力機制（attention function）概念上就是拿一個查詢（query）去跟一組 key-values 做運算，最後產生一個輸出。
    只是我們會利用矩陣運算同時讓多個查詢跟一組 key-values 做運算，最大化計算效率。
    :param q:
    :param k:
    :param v:
    :param mask:
    :return:
        values: 代表注意力機制的結果
        attention: 代表句子 q 裡頭每個子詞對句子 k 裡頭的每個子詞的注意權重
    """
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.words.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.words.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        """
        :param x:
        :param mask:
        :param return_attention:
        :return: (batch_size, seq_len_q, d_model)
        """
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


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
        self.save_hyperparameters()
        self.batch_size = 0
        self._device = device
        self.grads_writer = SummaryWriter('logs/grads')
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

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, batch):
        output = self.model(batch)
        return output

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        # optimizer = optim.Adam(self.parameters(), lr=0.0005)
        optimizer = optim.AdamW(self.parameters())
        # optimizer = optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=50, max_iters=80
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self._calculate_loss(outputs, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        outputs = self(batch)
        loss = self._calculate_loss(outputs, batch_idx)
        self.write_grads(batch_idx)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self._calculate_loss(outputs, batch_idx)
        self.log("test_loss", loss)

    def _calculate_loss(self, batch, batch_idx):
        loss = self.criterion(batch)
        return loss

    def write_grads(self, epoch):
        grads = []
        for param in self.model.parameters():
            grads.append(param.grad)
        # 將梯度轉換為張量
        grads_tensor = torch.stack(grads)
        self.grads_writer.add_embedding(grads_tensor, global_step=epoch)

    def draw_grads(self):
        # 獲取模型的參數梯度
        gradients = [param.grad for param in self.model.parameters()]
        # 使用 Matplotlib 繪製梯度 3D 圖
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(gradients)
        plt.show()

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


def load_model(model_type, model_args, checkpoint_path="./output", model_name=None):
    model_name = model_type.__name__ if model_name is None else model_name
    pretrained_filename = os.path.join(checkpoint_path, f"{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        info(f"Found pretrained model, loading {model_name}...")
        model = model_type(**model_args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        info(f"{device=}")
        return model.load_from_checkpoint(pretrained_filename).to(device)
    info_error(f"Not found pretrained model, {model_name}...")
    return None


def start_train(model_type,
                model_args,
                dataset,
                batch_size=1,
                max_epochs=10,
                resume_train=False,
                checkpoint_path="./output",
                model_name=None,
                device=None,
                ):
    model_name = model_type.__name__ if model_name is None else model_name
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir = os.path.join(checkpoint_path, model_name)
    os.makedirs(root_dir, exist_ok=True)
    # last_checkpoint_file_path = f'./output/{model_name}.ckpt'
    last_checkpoint_file_path = get_last_train_ckpt_file(f'./output/{model_name}')
    if not is_file_exists(last_checkpoint_file_path):
        last_checkpoint_file_path = None
    last_checkpoint_file_path = None if resume_train is False else last_checkpoint_file_path
    save_weights_only = True if resume_train is False else False
    info(f" {save_weights_only=} {last_checkpoint_file_path=}")
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        # callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
        callbacks=[ModelCheckpoint(filename="{epoch}-{train_loss:.5f}",
                                   save_weights_only=save_weights_only, mode="min", monitor="train_loss")],
        # gpus=1 if str(device).startswith("cuda") else 0,
        accelerator='gpu', devices=1,
        max_epochs=max_epochs,
        gradient_clip_val=10,
        resume_from_checkpoint=last_checkpoint_file_path
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    pretrained_filename = os.path.join(checkpoint_path, f"{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        info(f"Found pretrained model, loading {model_name}...")
        model = model_type.load_from_checkpoint(pretrained_filename, **model_args)
    else:
        model = model_type(**model_args)

    model._device = device
    model.init_dataloader(dataset, batch_size)

    info(f" start {model_name} model")
    # trainer.fit(model, train_loader, val_loader)
    trainer.fit(model)

    # Test best model on validation and test set
    val_loader = model.val_dataloader()
    test_loader = model.test_dataloader()
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    copy_last_ckpt(model_name)
    # result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}
    # model = model.to(device)
    return model  # , result


class LightningLogsIterator:
    def __init__(self, lightning_logs_path):
        self.lightning_logs_path = lightning_logs_path + "/lightning_logs"

    def __iter__(self):
        for folder in get_directory_list_by_pattern(self.lightning_logs_path, r'version_\d+'):
            for ckpt in get_file_list_by_pattern(folder + '/checkpoints', r'.+\.ckpt'):
                yield ckpt


def query_train_ckpts(ckpt_root_path='./output/BpeTranslator'):
    loss = re.compile(r'epoch=\d+\-train_loss=(\d+\.\d+)')
    for ckpt in LightningLogsIterator(ckpt_root_path):
        match = loss.search(ckpt)
        if match:
            yield match.group(1), ckpt


def get_last_train_ckpt_file(ckpt_root_path):
    ckpt_list = [x for x in query_train_ckpts(ckpt_root_path)]
    if not ckpt_list:
        return None
    _, ckpt = min(ckpt_list, key=lambda tup: tup[0])
    return ckpt


def copy_last_ckpt(model_type):
    if isinstance(model_type, str):
        model_name = model_type
    else:
        model_name = model_type.__name__
    ckpt = get_last_train_ckpt_file(f"./output/{model_name}")
    if ckpt is None:
        return
    print(f"{model_name} {ckpt=}")
    shutil.copy(ckpt, './output/%s.ckpt' % model_name)
