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
        self._token_to_idx = {}
        self._idx_to_token = {}
        self.recreate_dict(token_to_idx)

    def recreate_dict(self, token_to_idx):
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
            raise KeyError(f"The index {type(index)} {index} is not in the Vocabulary")
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size={})>".format(len(self))

    def __len__(self):
        return len(self._token_to_idx)


class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk='<UNK>',
                 mask='<MASK>', pad='<PAD>',
                 sos='<SOS>', eos="<EOS>"):
        super(SequenceVocabulary, self).__init__(token_to_idx)

        self.MASK = mask
        self.UNK = unk
        self.PAD = pad
        self.SOS = sos
        self.EOS = eos

        self.MASK_index = self.add_token(self.MASK)
        self.UNK_index = self.add_token(self.UNK)
        self.PAD_index = self.add_token(self.PAD)
        self.SOS_index = self.add_token(self.SOS)
        self.EOS_index = self.add_token(self.EOS)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({
            '<UNK>': self.UNK,
            '<MASK>': self.MASK,
            '<PAD>': self.PAD,
            '<SOS>': self.SOS,
            '<EOS>': self.EOS,
        })
        return contents

    def lookup_token(self, token: str) -> int:
        if self.UNK_index >= 0:
            return self._token_to_idx.get(token, self.UNK_index)
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

    def from_serializable(self, token_to_idx):
        self.vocab.recreate_dict(token_to_idx)

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

    def decode_value_list(self, index_list: list[int], isShow: bool=False) -> str:
        words = []
        max_index = len(index_list)
        idx = 0
        while idx < max_index:
            value = index_list[idx]
            if isShow:
                match value:
                    case value if isinstance(value, Enum):
                        idx += 1
                        continue
                    case self.vocab.SOS_index:
                        words += self.vocab.SOS
                        idx += 1
                        continue
                    case self.vocab.EOS_index:
                        words += self.vocab.EOS
                        idx += 1
                        continue
                    case self.vocab.PAD_index:
                        words += self.vocab.PAD
                        idx += 1
                        continue
            else:
                if isinstance(value, Enum) or \
                        value in [self.vocab.SOS_index, self.vocab.EOS_index, self.vocab.PAD_index]:
                    idx += 1
                    continue
            if (idx+1) >= max_index:
                break
            capitalization = index_list[idx+1]
            if isShow:
                match capitalization:
                    case self.vocab.SOS_index:
                        idx += 2
                        words += self.vocab.SOS
                        continue
                    case self.vocab.EOS_index:
                        idx += 2
                        words += self.vocab.EOS
                        continue
                    case self.vocab.PAD_index:
                        idx += 2
                        words += self.vocab.PAD_index
                        continue
            else:
                if isinstance(value, Enum) or \
                        value in [self.vocab.SOS_index, self.vocab.EOS_index, self.vocab.PAD_index]:
                    idx += 2
                    continue
            words += self.decode_index([value, capitalization])
            idx += 2
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

