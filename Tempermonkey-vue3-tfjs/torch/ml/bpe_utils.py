import collections
import re
from collections import Counter

from utils.data_utils import create_char2index_map, create_index2char_map


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
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, vocab):
    vocab_out = {}
    new_symbol = ''.join(pair)
    bigram = ' '.join(pair)
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab:
        new_word = p.sub(new_symbol, word)
        vocab_out[new_word] = vocab[word]
    return vocab_out


def get_tokens_from_vocab(vocab):
    tokens_frequencies = collections.defaultdict(int)
    vocab_tokenization = {}
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens_frequencies[token] += freq
        vocab_tokenization[''.join(word_tokens)] = word_tokens
    return tokens_frequencies, vocab_tokenization


class BPE:
    def __init__(self):
        self.tokens_frequencies = {}
        self.vocab_tokenization = []
        self.token2index = {}
        self.index2token = {}

    def build(self, tokens):
        tokens = [token.replace(' ', '</s>') for token in tokens]
        vocab = collections.defaultdict(int)
        for token in tokens:
            vocab[' '.join(list(token)) + ' </w>'] += 1
        self.run_merge_vocab(vocab)
        sorted_tokens_tuple = sorted(self.tokens_frequencies.items(),
                                     key=lambda item: (measure_token_length(item[0]), item[1]), reverse=True)
        tokens = [token for (token, freq) in sorted_tokens_tuple] + ['</u>']
        self.token2index = create_char2index_map(tokens, 1)
        self.index2token = create_index2char_map(tokens, 1)

    def run_merge_vocab(self, vocab):
        num_merges = 5
        for i in range(num_merges):
            pairs = get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            if pairs[best] == 1:
                break
            vocab = merge_vocab(best, vocab)
            tokens_frequencies, vocab_tokenization = get_tokens_from_vocab(vocab)
            self.tokens_frequencies = tokens_frequencies
            self.vocab_tokenization = vocab_tokenization

    def encode_to_tokens(self, word):
        if word in self.vocab_tokenization:
            return self.vocab_tokenization[word]
        return tokenize_word(text=word, sorted_tokens=self.token2index, unknown_token='</u>')

    def encode(self, word):
        tokens = self.encode_to_tokens(word)
        return [self.token2index[token] for token in tokens]

    def decode(self, values):
        tokens = [self.index2token[value] for value in values]
        tokens = [token.replace('</s>', ' ') for token in tokens]
        return ''.join(tokens)

