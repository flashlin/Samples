import collections
import os
import re
from collections import Counter

from common.io import info
from utils.data_utils import create_char2index_map, create_index2char_map


def map_data_path(relative_file):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f"data/{relative_file}")


def tokenize_word(text, sorted_tokens, unknown_token='</u>'):
    tokens = [char.replace(' ', '</s>') for char in text]
    if not tokens:
        return []
    record_tokens = {}
    new_tokens = []
    n = 0
    while n < len(tokens):
        merged = False
        for cut_len in range(len(tokens), 0, -1):
            token = ''.join(tokens[n:n + cut_len])
            if token in sorted_tokens:
                merged = True
                record_tokens[token] = True
                new_tokens.append(token)
                n += cut_len
                break
        if not merged:
            new_tokens.append(tokens[n])
            n += 1
        else:
            continue
    new_tokens2 = []
    for n in range(len(new_tokens)):
        token = new_tokens[n]
        if token not in record_tokens:
            new_tokens2.append(unknown_token)
        else:
            new_tokens2.append(token)
    return new_tokens2


def measure_token_length(token):
    chars = ['</w>', '</s>']
    for char in chars:
        token = token.replace(char, ' ')
    return len(token)


def generate_bpe_vocabulary_and_merges(tokens, num_merges=10):
    # Initialize the vocabulary and the merges
    vocabulary = {}
    merges = []

    # Create a counter for the tokens
    token_counts = Counter(tokens)

    # Initialize the current vocabulary size
    vocabulary_size = 0

    # Iterate over the tokens
    for token, count in token_counts.items():
        # If the token doesn't exist in the vocabulary, add it
        if token not in vocabulary:
            vocabulary[token] = {
                'count': count,
                'idx': vocabulary_size
            }
            vocabulary_size += 1

    merge_index = 0
    for token, count in token_counts.items():
        if count == 1:
            continue

        current_merge = [token]
        for remaining_token in token_counts.keys():
            # Skip the current token
            if remaining_token == token:
                continue

            # Check if the current merge ends with the remaining token
            if current_merge[-1].endswith(remaining_token):
                current_merge.append(remaining_token)

        # Check if the current merge has more than one element
        if len(current_merge) > 1:
            # Add the current merge to the list of merges
            merges.append((current_merge[0], current_merge[1]))

            # Increment the current merge index
            merge_index += 1

            # Stop if the required number of merges has been reached
            if merge_index >= num_merges:
                break

    return vocabulary, merges


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for (symbols, freq) in vocab:
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def search_and_merge(alist, pair):
    alist_len = len(alist)
    str_len = len(pair)
    result = []
    pair_symbol = ''.join(pair)
    n = 0
    while n < alist_len:
        part = ''.join(alist[n: n+str_len])
        if part == pair_symbol:
            result.append(part)
            n += str_len
            continue
        result.append(alist[n])
        n += 1
    return result



def merge_vocab(pair, vocab):
    vocab_out = StrListDict()
    new_symbol = ''.join(pair)
    # bigram = ' '.join(pair)
    # p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word, count in vocab:
        new_word = search_and_merge(word, pair)
        vocab_out[new_word] = vocab[word]
    return vocab_out


def get_tokens_from_vocab(vocab):
    vocab_tokenization = {}
    for word, freq in vocab:
        vocab_tokenization[''.join(word)] = word
    return vocab_tokenization


def dump(token_freq):
    for token, count in token_freq:
        print(f"{token}: {count}")


class StrListDict(collections.UserDict):
    @staticmethod
    def get_key(alist):
        def replace_char(ch):
            if ch == '</s>':
                return ' '
            if ch == '</w>':
                return ''
            return ch
        chars = [replace_char(char) for char in alist]
        return ''.join(chars)

    def __setitem__(self, alist, value):
        # 檢查鍵是否為字符串列表
        if self.is_key(alist):
            key = ' '.join(alist)
            if isinstance(value, int):
                self.data[key] = (alist, value)
            else:
                raise ValueError("Value must be an integer")
        else:
            raise KeyError("Key must be a list of strings")

    def __getitem__(self, alist):
        if not self.is_key(alist):
            raise KeyError(f"'{alist}' Key must be a list of string")
        key = ' '.join(alist)
        if key in self.data:
            return self.data[key][1]
        else:
            raise KeyError(f"'{key}' Key not found")

    def __iter__(self):
        # 將字典鍵值對按鍵的字符串列表長度排序
        items = sorted(self.data.items(), key=lambda x: -len(self.get_key(x[0])))
        result = []
        for (key, a_tuple) in items:
            result.append(a_tuple)
        return iter(result)

    def __contains__(self, alist):
        key = ' '.join(alist)
        return key in self.data

    def increase_count(self, alist):
        if alist in self:
            self[alist] += 1
        else:
            self[alist] = 1

    @staticmethod
    def is_key(alist):
        return isinstance(alist, list) and all(isinstance(x, str) for x in alist)


class BPE:
    def __init__(self):
        self.char_frequencies = StrListDict()
        self.vocab_tokenization = []
        self.token2index = {}
        self.index2token = {}

    def dump(self):
        info(f"char_freq")
        for char, count in self.char_frequencies:
            print(f"{char}: {count}")
        print(f"{self.vocab_tokenization=}")

    def add_word_list(self, word_list):
        for word in word_list:
            self.add_word(word)

    def build(self):
        vocab = self.merge_vocab(self.char_frequencies)
        word = [''.join(char) for (char, freq) in vocab] + ['</u>']
        self.token2index = create_char2index_map(word, 1)
        self.index2token = create_index2char_map(word, 1)

    def add_word(self, word):
        word = [char.replace(' ', '</s>') for char in word] + ['</w>']
        vocab = self.char_frequencies
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1

    def merge_vocab(self, vocab):
        num_merges = 100
        for i in range(num_merges):
            pairs = get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            # if pairs[best] == 1:
            #     break
            info(f" {best=}")
            vocab = merge_vocab(best, vocab)
            dump(vocab)
        vocab_tokenization = get_tokens_from_vocab(vocab)
        self.char_frequencies = vocab
        self.vocab_tokenization = vocab_tokenization
        return vocab

    def encode_to_tokens(self, word):
        word += '</w>'
        if word in self.vocab_tokenization:
            return self.vocab_tokenization[word]
        return tokenize_word(text=word, sorted_tokens=self.token2index, unknown_token='</u>')

    def encode(self, word):
        tokens = self.encode_to_tokens(word)
        return [self.token2index[token] for token in tokens]

    def decode(self, values):
        def remove_w(t):
            if t.endswith('</w>'):
                return t[:-4]
            return t
        tokens = [self.index2token[value] for value in values]
        tokens = [token.replace('</s>', ' ') for token in tokens]
        tokens = [remove_w(token) for token in tokens]
        return ''.join(tokens)


class TextEncoding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bpe = BPE()
        # self.read_cvc_file()

    def encode(self, text):
        values = []
        words = self.tokenizer.tokenize_to_words(text)
        for word in words:
            values += self.bpe.encode(word)
        return values

    def decode(self, text_values):
        return self.bpe.decode(text_values)

    def build(self):
        self.bpe.build()
        #self.bpe.dump()

    def read_text_file(self, text_file):
        words = self.tokenizer.read_text_file_to_words(text_file)
        self.bpe.add_word_list(words)

    def read_cvc_file(self):
        words = []
        cvc_file = map_data_path('cvc.txt')
        with open(cvc_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                for word in line.split(' '):
                    word = word.strip()
                    words.append(word)
        self.bpe.add_word_list(words)
