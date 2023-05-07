import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tsql_tokenizr import tsql_tokenize
from vocabulary_utils import WordVocabulary
from itertools import cycle, islice


class LabelTransform(object):
    """
    將不同長度的答案拓展到相同長度，以便訓練模型
    """

    def __init__(self, max_size: int, pad_value: list[int]):
        self.max_size = max_size
        self.pad_value = pad_value

    def __call__(self, value_seq: list[int]):
        assert len(value_seq) < self.max_size, f"please ensure sentence len {len(value_seq)} < {self.max_size}"
        # label = np.pad(label, (0, (self.max_size - label.shape[0])), mode='constant', constant_values=self.pad_value)
        result = list(islice(cycle(self.pad_value), self.max_size - len(value_seq)))
        new_label = value_seq + result
        return new_label


class WordDataset(Dataset):
    def __init__(self, text_list, max_output_len=40):
        self.max_output_len = max_output_len
        self.tokenize = tsql_tokenize
        self.vocab = WordVocabulary()
        self.vocab.vocab.add_many(['<BOS>', '<EOS>', '<UNK>'])
        self.data = text_list
        self.vocab_size = 50000
        self.transform = LabelTransform(max_output_len, self.vocab.encode_word('<PAD>'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentences = self.data[index]
        input_text = sentences['input']
        target_text = sentences['target']
        input_seq = self.text_to_values(input_text)
        target_seq = self.text_to_values(target_text)
        # 用 <PAD> 將句子補到相同長度
        input_seq, target_seq = self.transform(input_seq), self.transform(target_seq)
        input_seq, target_seq = torch.LongTensor(input_seq), torch.LongTensor(target_seq)
        return input_seq, target_seq

    def text_to_values(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        words = [token.text for token in tokens]
        value_seq = self.vocab.encode_many_words(words)
        bos = self.vocab.encode_word('<BOS>')
        eos = self.vocab.encode_word('<EOS>')
        return bos + value_seq + eos

    
