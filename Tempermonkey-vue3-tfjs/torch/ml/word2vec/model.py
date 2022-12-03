import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
from functools import partial

from common.io import info
from ml.lit import BaseLightning
from ml.translate_net import get_translate_file_iter

EMBED_DIMENSION = 300
EMBED_MAX_NORM = 1


class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


CBOW_N_WORDS = 4
MAX_SEQUENCE_LENGTH = 256
SKIPGRAM_N_WORDS = 4


def collate_cbow(batch, text_pipeline):
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx: (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x


def collate_skipgram(batch, text_pipeline):
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - SKIPGRAM_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx: (idx + SKIPGRAM_N_WORDS * 2 + 1)]
            input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)
            outputs = token_id_sequence

            for output in outputs:
                batch_input.append(input_)
                batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


class Word2Vec(BaseLightning):
    def __init__(self, vocab_size):
        super().__init__()
        self.cbow_model = CBOW_Model(vocab_size)
        self.skip_gram_model = CBOW_Model(vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        inputs, labels = batch
        outputs1 = self.cbow_model(inputs)
        outputs2 = self.skip_gram_model(inputs)
        return outputs1, outputs2, labels

    def _calculate_loss(self, batch, mode="train"):
        (outputs1, outputs2, labels), batch_idx = batch
        loss1 = self.loss_fn(outputs1, labels)
        loss2 = self.loss_fn(outputs2, labels)
        loss = loss1 + loss2
        self.log("%s_loss" % mode, loss)
        return loss

    def infer(self, vocab, word):
        self.cbow_model.eval()
        self.skip_gram_model.eval()
        value = vocab.encode(word, False)
        value = torch.tensor([value], dtype=torch.long)
        vec1 = self.cbow_model(value)
        vec2 = self.skip_gram_model(value)
        return vec1 * vec2


class TranslateFileDataset(Dataset):
    def __init__(self, translation_file_path, vocab):
        self.vocab = vocab
        self.text_list = []
        for src, tgt in get_translate_file_iter(translation_file_path):
            self.text_list.append(src)
            self.text_list.append(tgt)

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        src = self.text_list[idx]
        # src = self.vocab.encode(src)
        return src

    def create_dataloader(self, batch_size=32):
        train_size = int(0.8 * len(self))
        val_size = len(self) - train_size
        train_data, val_data = random_split(self, [train_size, val_size])
        train_loader = self.create_data_loader(train_data, batch_size=batch_size)
        val_loader = self.create_data_loader(val_data, batch_size=batch_size)
        return train_loader, val_loader

    def create_data_loader(self, data_set, batch_size):
        collate_fn = collate_cbow
        text_pipeline = lambda x: self.vocab.encode(x, False)
        return DataLoader(dataset=data_set, batch_size=batch_size,
                          collate_fn=partial(collate_fn, text_pipeline=text_pipeline))

