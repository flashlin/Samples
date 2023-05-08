# https://github.com/yejh123/Transformer/blob/master/code/train.py
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from data import WordDataset
from config import Config
from transformer_model import Transformer

train_text_list = [
    {'input': 'select 1', 'target': 'SELECT 1'},
    {'input': 'select id from c1', 'target': 'SELECT id FROM c1 WITH(NOLOCK)'},
]

config = Config()
train_dataset = WordDataset(train_text_list)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
# train_iter = infinite_iter(train_loader)

transformer_model = Transformer(config.vocab_size,
                                config.max_output_len,
                                config.vocab_size,
                                config.max_output_len,
                                num_layers=config.n_layers,
                                model_dim=config.model_dim,
                                num_heads=config.num_heads,
                                ffn_dim=config.ffn_dim,
                                dropout=config.dropout,
                                ).to(config.device)

