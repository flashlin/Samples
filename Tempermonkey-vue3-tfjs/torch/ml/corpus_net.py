import collections
import os
import re

import torch
import torch.nn as nn

from ml.lit import BaseLightning
from ml.text_classifier_net import ProgramLangVocab
from utils.stream import Token
from collections import defaultdict
from gensim import corpora
import sentencepiece as spm
from torchtext.data.functional import generate_sp_model, load_sp_model, sentencepiece_numericalizer

text_corpus = [
    "from tb1 in customer select tb1.name",
    "from tb2 in user join tb3 in home on tb2.id equals tb3.id select new {tb2.id, tb3.addr}",
    "from tb1 in customer select new {tb1.id, price = tb1.p + 123}",
]


def default_bpe_folder():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/")


def get_word_stats(word_freq):
    pairs = collections.defaultdict(int)
    for word, freq in word_freq.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


class CharLevelBpeVocab:
    def __init__(self, vocab):
        self.vocab = vocab
        self.word_freq = collections.defaultdict(int)
        self.pairs = None

    def build_from_alist(self, alist):
        for word in alist:
            self.build_word(word)

    def build_word(self, word):
        word_freq = self.word_freq
        a_char_list = self.word_to_list(word)
        word_freq[' '.join(a_char_list) + ' </w>'] += 1
        self.word_freq = word_freq
        return word_freq

    def encode(self, word: str) -> list[int]:
        if word in self.word_freq:
            return [self.word_freq[word]]

        words = [(word[:i], word[i:]) for i in range(len(word), 0, -1)]
        words = [i for i in words if i[0] and i[1]]

        for pair in self.pairs:
            if pair in self.word_freq:
                bpe_word = pair.replace(" ", "")
                new_words = []
                for word in words:
                    if word[0] == pair:
                        new_words.append((bpe_word, word[1]))
                    elif word[1] == pair:
                        new_words.append((word[0], bpe_word))
                    else:
                        new_words.append(word)
                words = new_words

        words = ["".join(word) for word in words if word[1]]
        return words

    @staticmethod
    def merge_vocab(pair, word_freq):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in word_freq:
            w_out = p.sub(' '.join(pair), word)
            v_out[w_out] = word_freq[word]
        return v_out

    def merge(self):
        num_merges = 5
        word_freq = self.word_freq
        for i in range(num_merges):
            pairs = get_word_stats(word_freq)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            word_freq = self.merge_vocab(best, word_freq)
        self.word_freq = word_freq
        self.pairs = pairs

    @staticmethod
    def word_to_list(word):
        return [char if char != ' ' else '</sp>' for char in list(word)]


class ByteLevelBpeVocab:
    def __init__(self, vocab):
        self.vocab = vocab
        self.pairs = torch.empty((0, 2), dtype=torch.long)

    def build_pairs(self, word):
        vocab = self.vocab
        pairs = self.pairs
        for n, char in enumerate(word[:-1]):
            # 將當前字符和下一個字符組成一個 pair
            pair = torch.tensor([vocab.char2int[char], vocab.char2int[word[n + 1]]], dtype=torch.long)
            pairs = torch.cat([pairs, pair.unsqueeze(0)], dim=0)
        return pairs

    def encode(self, token: str) -> list[int]:
        vocab = self.vocab
        if token in vocab.char2int:
            return vocab.char2int[token]

        pairs = self.pairs

        words = [(token[:i], token[i:]) for i in range(len(token), 0, -1)]
        words = [char for char in words if char[0] and char[1]]

        for pair in pairs:
            if pair in vocab.char2int:
                bpe_word = pair.replace(" ", "")
                new_words = []
                for word in words:
                    if word[0] == pair:
                        new_words.append((bpe_word, word[1]))
                    elif word[1] == pair:
                        new_words.append((word[0], bpe_word))
                    else:
                        new_words.append(word)
                words = new_words

        words = ["".join(word) for word in words if word[1]]
        return words


class BpeVocab:
    def __init__(self, vocab: ProgramLangVocab = ProgramLangVocab()):
        bpe_model_name = 'program_lang'
        self.vocab = vocab
        self.bpe_model_name = bpe_model_name
        self.sp = None

    def create_train_csv_file(self, list_of_text):
        with open('train_data/bpe.csv', 'w', encoding='utf-8') as f:
            for text in list_of_text:
                tokens = self.vocab.parse_to_tokens(text)
                for token in tokens:
                    if token.text.strip() not in ['', '\n']:
                        if token.type not in [Token.Symbol, Token.Operator]:
                            f.write(token.text + '\n')
            f.write('1234567890\n')
            f.write('abcdefghijklmnopqrstuvwxyz\n')
            f.write('ABCDEFGHIJKLMNOPQRSTUVWXYZ\n')
            f.write('~!@#$%^&*()_+`_+{}|{}|:";\'<>?,./\n')

    def train(self):
        # model_type： 模型的類型，包括 unigram, bpe, char, word
        # model_prefix: 保存模型和詞彙表的文件的前缀
        # vocab_size: 最小為 70 字母, 理論要大於 70 才能賦予 word 意義
        generate_sp_model('train_data/bpe.csv', vocab_size=70 + 20, model_type='bpe', model_prefix=self.bpe_model_name)

    def load(self):
        sp_model = load_sp_model(f"{self.bpe_model_name}.model")
        self.sp = sentencepiece_numericalizer(sp_model)

    def encode_tokens(self, tokens):
        # return list(self.sp_id_generator(tokens))
        return self.sp.EncodeAsPieces(tokens)

    def decode_values(self, values):
        sp = spm.SentencePieceProcessor(model_file=f'{self.bpe_model_name}.model')
        return sp.decode(values)


# pip install sentence-transformers


class CorpusVocab:
    def __init__(self, vocab):
        self.vocab = vocab
        self.dictionary = None

    def tokenize(self, list_of_text):
        frequency = defaultdict(int)
        for text in list_of_text:
            for token in text:
                frequency[token] += 1
        processed_corpus = [[token for token in text if frequency[token] > 1] for text in list_of_text]
        dictionary = corpora.Dictionary(processed_corpus)
        # print(dictionary.token2id)
        self.dictionary = dictionary

    def to_vector(self, new_doc):
        tokens = self.vocab.encode_to_tokens(new_doc)
        new_vec = self.dictionary.doc2bow(tokens)


class LitTransformer2(BaseLightning):
    def __init__(self):
        super().__init__()
        vocab = ProgramLangVocab()
        self.model = TextClassifier(vocab=vocab, num_class=vocab.get_size())
        # CrossEntropyLoss((src_len, n_classes), (tgt_len))
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.padding_idx)

    def forward(self, batch):
        device = next(self.parameters()).device
        src, src_lens, tgt, tgt_lens = batch
        src = pad_batch_sequence(src, 200).type(torch.long).to(device)
        tgt = pad_batch_sequence(tgt, 200).type(torch.long).to(device)
        return self.model(src, tgt), tgt

    def _calculate_loss(self, batch, batch_idx):
        x_hat, y = batch
        x_hat = x_hat.squeeze(0)
        y_true = y.squeeze(0)
        return self.loss_fn(x_hat, y_true)

    def infer(self, text):
        device = self.get_device()
        text_to_indices = self.vocab.encode_to_tokens(text)

        src = torch.tensor([text_to_indices]).type(torch.long).to(device)
        src = pad_batch_sequence(src, 200).type(torch.long).to(device)
        self.model.eval()
        logits = self.model(src, src)
        logits = logits.squeeze(0).argmax(1).tolist()
        return self.vocab.decode(logits)

# Generate the vocabulary and the merges for the text
# vocabulary, merges = generate_bpe_vocabulary_and_merges("the lazy dog laid low on the bed")
