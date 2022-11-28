from torch import nn
from torch.utils.data import Dataset

from ml.lit import BaseLightning
from ml.seq2seq_model import Seq2SeqTransformer
from ml.bpe_tokenizer import SimpleTokenizer
from preprocess_data import TranslationDataset, TranslationFileTextIterator, int_list_to_str
from utils.linq_tokenizr import linq_tokenize, linq_encode, LINQ_VOCAB_SIZE
from utils.stream import Token
from utils.tokenizr import BOS_TOKEN_VALUE, EOS_TOKEN_VALUE, PAD_TOKEN_VALUE
from utils.data_utils import sort_desc, create_char2index_map, create_index2char_map
from utils.tsql_tokenizr import tsql_tokenize, tsql_encode, TSQL_VOCAB_SIZE, tsql_decode
from dataclasses import dataclass, field
from typing import Callable
from collections import UserDict, Iterable


class TranslationFileTokenIterator:
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        for src, tgt in TranslationFileTextIterator(self.file_path):
            src_tokens = linq_tokenize(src)
            tgt_tokens = tsql_tokenize(tgt)
            yield src_tokens, tgt_tokens


class TokenType:
    POS = '<pos>'
    IDENTIFIER = '<identifier>'
    STRING = '<string>'
    NUMBER = '<number>'
    OPERATOR = '<operator>'
    SYMBOL = '<symbol>'
    KEYWORD = '<keyword>'
    SPACE = '<space>'


TokenType_Char2index = create_char2index_map(
    [
        TokenType.POS,
        TokenType.IDENTIFIER, TokenType.STRING, TokenType.NUMBER,
        TokenType.OPERATOR, TokenType.SYMBOL, TokenType.KEYWORD,
        TokenType.SPACE
    ])


class TokenTypeIndex:
    POS = TokenType_Char2index[TokenType.POS]
    IDENTIFIER = TokenType_Char2index[TokenType.IDENTIFIER]
    STRING = TokenType_Char2index[TokenType.STRING]
    NUMBER = TokenType_Char2index[TokenType.NUMBER]
    OPERATOR = TokenType_Char2index[TokenType.OPERATOR]
    SYMBOL = TokenType_Char2index[TokenType.SYMBOL]
    KEYWORD = TokenType_Char2index[TokenType.KEYWORD]
    SPACE = TokenType_Char2index[TokenType.SPACE]


class TokenTypeMap:
    Types = sort_desc([
        TokenType.IDENTIFIER,
        TokenType.STRING,
        TokenType.NUMBER,
        TokenType.OPERATOR,
        TokenType.SYMBOL,
        TokenType.KEYWORD,
        TokenType.SPACE
    ])
    Operators = sort_desc([
        '>', '<', '>=', '<=', '==', '!=',
        '+=', '-=', '+', '-', '*', '/', '^', '||', '&&',
        '(', ')', '{', '}', ',', '=',
    ])
    Keywords = sort_desc([
        'select',
        'from',
        'in',
        'on',
        'new',
        'equals',
        'SELECT',
        'AS',
        'FROM',
        'NOLOCK',
        'WITH',
        'ON'
    ])

    def __init__(self):
        super().__init__()
        data = TokenTypeMap.Types + TokenTypeMap.Operators + TokenTypeMap.Keywords
        self.char2index = create_char2index_map(data)
        self.index2char = create_index2char_map(data)

    def get_value(self, char):
        self.char2index[char]

    def get_token_value(self, idx, token: Token):
        if token.type == Token.Identifier:
            return [TokenTypeIndex.IDENTIFIER, idx]
        if token.type == Token.String:
            return [TokenTypeIndex.STRING, idx]
        if token.type == Token.Number:
            return [TokenTypeIndex.NUMBER, idx]
        if token.type == Token.Operator:
            return [TokenTypeIndex.OPERATOR, self.get_value(token.text)]
        if token.type == Token.Symbol:
            return [TokenTypeIndex.SYMBOL, self.get_value(token.text)]
        if token.type == Token.Keyword:
            return [TokenTypeIndex.KEYWORD, self.get_value(token.text)]
        raise Exception(f'{token.type} {token=}')


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


class TranslationFileEncodeIterator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.token_map = TokenTypeMap()

    def __iter__(self):
        for src_tokens, tgt_tokens in TranslationFileTokenIterator(self.file_path):
            src = flatten([self.token_map.get_token_value(idx, token) for idx, token in enumerate(src_tokens)])
            tgt = flatten([self.token_map.get_token_value(idx, token) for idx, token in enumerate(tgt_tokens)])
            yield src, tgt


def write_train_csv_file():
    with open('./output/linq-sample.csv', "w", encoding='UTF-8') as csv:
        csv.write('src\ttgt\n')
        for src, tgt in TranslationFileEncodeIterator('../data/linq-sample.txt'):
            csv.write(int_list_to_str(src))
            csv.write('\t')
            csv.write(int_list_to_str(tgt))
            csv.write('\n')


@dataclass(frozen=True)
class TranslateOptions:
    src_vocab_size: int
    tgt_vocab_size: int
    bos_idx: int
    eos_idx: int
    padding_idx: int
    decode_fn: Callable[[list[int]], str]
    train_dataset: Dataset


bpe_translate_options = TranslateOptions(
    src_vocab_size=LINQ_VOCAB_SIZE,
    tgt_vocab_size=TSQL_VOCAB_SIZE,
    bos_idx=BOS_TOKEN_VALUE,
    eos_idx=EOS_TOKEN_VALUE,
    padding_idx=PAD_TOKEN_VALUE,
    decode_fn=tsql_decode,
    train_dataset=lambda: TranslationDataset("./output/linq-sample.csv", PAD_TOKEN_VALUE)
)


class TokenTypeTranslator(BaseLightning):
    def __init__(self, options=bpe_translate_options):
        super().__init__()
        self.options = options
        self.model = Seq2SeqTransformer(options.src_vocab_size,
                                        options.tgt_vocab_size,
                                        bos_idx=options.bos_idx,
                                        eos_idx=options.eos_idx,
                                        padding_idx=options.padding_idx)
        self.criterion = nn.CrossEntropyLoss()
        self.init_dataloader(options.train_dataset(), 1)

    @staticmethod
    def prepare_train_data():
        write_train_csv_file()

    def forward(self, batch):
        enc_inputs, dec_inputs, dec_outputs = batch
        logits, enc_self_attns, dec_self_attns, dec_enc_attns = self.model(enc_inputs, dec_inputs)
        return logits, dec_outputs

    def _calculate_loss(self, data, mode="train"):
        (logits, dec_outputs), batch_idx = data
        loss = self.criterion(logits, dec_outputs.view(-1))
        self.log("%s_loss" % mode, loss)
        return loss

    def infer(self, text):
        tgt_values = self.model.inference(text)
        tgt_text = self.options.decode_fn(tgt_values)
        return tgt_text
