from tokenizr_utils import LETTERS
from enum import Enum
import re


class Vocabulary(object):
    """
    Class to process text and extract Vocabulary for mapping
    """

    def __init__(self, token_to_idx=None):
        """
        :param token_to_idx:  a pre-existing map of tokens to indices
        """
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {token: idx for idx, token in self._token_to_idx.items()}

    def to_serializable(self):
        """
        return a dictionary that can be serialized
        :return:
        """
        return {
            'token_to_idx': self._token_to_idx
        }

    @classmethod
    def from_serializable(cls, contents):
        """
        instantiates the Vocabulary from a serialized dictionary
        :param contents:
        :return:
        """
        return cls(**contents)

    def add_token(self, token: str) -> int:
        """
        Update mapping dicts based on the token
        :param token: str, the item to add into the Vocabulary
        :return:
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token

        return index

    def add_many(self, tokens: list[str]) -> list[int]:
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token: str) -> int:
        """
        Retrieve the index associated with the token or the UNK index if token isn't present
        :param token: (str) the token to look up
        :return: the index corresponding to the token
        Notes: 'unk_index' needs to be >= 0 (having been added into the Vocabulary)
        """
        return self._token_to_idx[token]

    def lookup_index(self, index: int) -> str:
        """
       Return the token associated with the index
       :param index: the index to lookup
       :return: the token corresponding to the index
       Raises KeyError if the index is not in the Vocabulary
       """
        if index not in self._idx_to_token:
            raise KeyError("The index {} is not in the Vocabulary".format(index))
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size={})>".format(len(self))

    def __len__(self):
        return len(self._token_to_idx)


class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>", end_seq_token="<END>"):
        super(SequenceVocabulary, self).__init__(token_to_idx)

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({
            'unk_token': self._unk_token,
            'mask_token': self._mask_token,
            "begin_seq_token": self._begin_seq_token,
            'end_seq_token': self._end_seq_token
        })
        return contents

    def lookup_token(self, token: str) -> int:
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx.get(token)


class WordType(Enum):
    Lower = 1
    Camel = 2
    Upper = 3
    Mix = 4


class WordVocabulary:
    def __init__(self):
        self.vocab = SequenceVocabulary()
        self.vocab.add_many(LETTERS)

    def to_serializable(self):
        return self.vocab.to_serializable()

    def encode_word(self, word: str) -> list[int]:
        word_list = self.split_text(word)
        result = []
        for new_word in word_list:
            word_lowered = new_word.lower()
            word_index = self.vocab.add_token(word_lowered)
            word_capitalization = self.get_word_type(new_word)
            concat_index = self.vocab.add_token('<+>')
            concat_capitalization = WordType.Lower
            result += [word_index, word_capitalization]
            result += [concat_index, concat_capitalization]
        result.pop()
        result.pop()
        return result

    def decode_index(self, index_list: list[int]) -> str:
        text = ""
        for idx in range(0, len(index_list) - 1, 2):
            word_index = index_list[idx]
            word_capitalization = index_list[idx+1]
            word = self.vocab.lookup_index(word_index)
            match word_capitalization:
                case WordType.Upper:
                    word = word.upper()
                case WordType.Camel:
                    word = word[0:1].upper() + word[1:]
            text += word
        return self.merge_concat_text(text)

    def encode_many_words(self, word_list: list[str]) -> list[int]:
        index_list = []
        for word in word_list:
            index_list += self.encode_word(word)
        return index_list

    def decode_index_list(self, index_list: list[int]) -> str:
        words = []
        for word_index, word_capitalization in zip(index_list[::2], index_list[1::2]):
            words += self.decode_index([word_index, word_capitalization])
        text = "".join(words)
        return self.merge_concat_text(text)

    @staticmethod
    def merge_concat_text(text: str) -> str:
        text = text.replace('<+>', '')
        return text

    @staticmethod
    def merge_concat_word_list(word_list: list[str]) -> list[str]:
        result = []
        concat_word = '<+>'
        i = 0
        while i < len(word_list):
            item = word_list[i]
            if item == concat_word:
                elem_merged = ''
                j = i + 1
                while j < len(word_list) and word_list[j] != concat_word:
                    elem_merged += word_list[j]
                    j += 1
                result.append(elem_merged)
                i = j
            else:
                result.append(item)
                i += 1
        return result

    @staticmethod
    def int_to_bits_str(num: int, content: str) -> str:
        bits = bin(num)[2:]
        bits = "0" * (len(content) - len(bits)) + bits if len(content) > len(bits) else bits
        return "".join([c.upper() if b == "1" else c for b, c in zip(bits, content)])

    @staticmethod
    def str_to_bits_int(word: str) -> int:
        bits = ''.join(['1' if c.isupper() else '0' for c in word])
        bits_int = int(bits, 2)
        return bits_int

    @staticmethod
    def get_word_type(word: str) -> WordType:
        if word.lower() == word:
            return WordType.Lower
        if word.isupper():
            return WordType.Upper
        first_letter = word[0]
        rest_letters = word[1:]
        if first_letter.isupper() and rest_letters.islower():
            return WordType.Camel
        return WordType.Mix

    @staticmethod
    def split_text(text: str) -> list[str]:
        pattern = r'[0-9]+|[A-Z][a-z]+\d*|[a-z]+\d*|[A-Z]+\d*|[^a-zA-Z0-9]+'
        result = re.findall(pattern, text)
        return result

