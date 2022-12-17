import collections
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SpmVocab:
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

def inputs_to_next_word_labels(inputs):
    """
    :param inputs: [1, 2, 3, 4]
    :return: [2, 3, 4, 0]
    """
    if len(inputs) > 1:
        return inputs[1:] + [0]
    else:
        return inputs


def inputs_batch_to_labels(inputs_batch):
    """
    :param inputs_batch: [[1,2,3],[4,5,6]]
    :return: [[2,3,0],[5,6,0]]
    """
    labels_batch = []
    for n in range(len(inputs_batch)):
        labels = inputs_to_next_word_labels(inputs_batch[n])
        labels_batch.append(labels)
    return labels_batch


class ContextVecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(2 * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs):
        # 取出詞彙的向量化表示
        embeddings = self.embeddings(inputs)

        # 計算詞彙的上下文出現機率
        out = self.linear1(embeddings)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.log_softmax(out, dim=1)
        return out

    def get_word_vectors(self):
        # 取出詞彙向量
        word_vectors = self.embeddings.weight.data
        return word_vectors

    def calculate_loss(self, x_hat, y_true):
        return self.loss_fn(x_hat, y_true)


class SequenceEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SequenceEncoderDecoder, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.decoder_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])

    def forward(self, x):
        """
        :param x: (seq_len)
        :return: 
        """
        encoded = []
        for i in range(x.size(0)):
            x_i = x[i]
            x_prev = x[i - 1] if i > 0 else torch.zeros_like(x_i)
            x_next = x[i + 1] if i < x.size(0) - 1 else torch.zeros_like(x_i)
            x_encoded = self.input_layer(x_i) + self.input_layer(x_prev) + self.input_layer(x_next)
            for hidden_layer in self.hidden_layers:
                x_encoded = hidden_layer(x_encoded)
            encoded.append(x_encoded)

        decoded = encoded[-1]
        for decoder_layer in self.decoder_layers:
            decoded = decoder_layer(decoded)
        return self.output_layer(decoded)


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
