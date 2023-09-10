import collections
import re

from labs.preprocess_data import TranslationFileTextIterator
from utils.linq_tokenizr import linq_tokenize
from utils.tsql_tokenizr import tsql_tokenize


def increase_count(text, word_dict):
    if text in word_dict:
        word_dict[text] += 1
    else:
        word_dict[text] = 1


def increase_tokens_count(tokens, word_dict):
    for token in tokens:
        increase_count(token.text, word_dict)


def generate_word_dict(sample_file):
    file_iter = TranslationFileTextIterator(sample_file)
    word_dict = {}
    for src, tgt in file_iter:
        src_tokens = linq_tokenize(src)
        tgt_tokens = tsql_tokenize(tgt)
        increase_tokens_count(src_tokens, word_dict)
        increase_tokens_count(tgt_tokens, word_dict)
    # for word, freq in word_dict.items():
    #     print(f"{word} {freq}")
    return word_dict


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    big_ram = re.escape(' '.join(pair))
    #p = re.compile(r'(?<!\S)' + big_ram + r'(?!\S)')
    p = re.compile(big_ram)
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def generate_vocab():
    word_dict = generate_word_dict('../../data/linq-sample.txt')
    vocab = {}
    for word in word_dict.keys():
        new_key = ' '.join([x for x in word] + ['</w>'])
        vocab[new_key] = word_dict[word]
    return vocab


def main():
    vocab = generate_vocab()
    num_merges = 10
    for i in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(best)


if __name__ == '__main__':
    main()
