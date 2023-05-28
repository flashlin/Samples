import unittest
from vocabulary_utils import WordVocabulary, WordType


class TestWordVocabulary(unittest.TestCase):
    def test_encode_decode(self):
        tk = WordVocabulary()
        index_list = tk.encode_many_words(['hello', ' ', 'NBAWorld'])
        text = tk.decode_value_list(index_list)
        assert text == "hello NBAWorld", text

    def test_encode_decode_word(self):
        tk = WordVocabulary()
        index_list = tk.encode_word('myNameNBA')
        word = tk.decode_index(index_list)
        assert word == "myNameNBA", word

    def test_split_text(self):
        word_list = WordVocabulary.split_text('hello1_123_1my_Name')
        assert word_list == ['hello1', '_', '123', '_', '1', 'my', '_',  'Name'], word_list

    def test_split_text2(self):
        word_list = WordVocabulary.split_text('hello1myName')
        assert word_list == ['hello1', 'my', 'Name'], word_list

    def test_split_text3(self):
        word_list = WordVocabulary.split_text('myIdName')
        assert word_list == ['my', 'Id', 'Name'], word_list

    def test_word_type(self):
        actual = WordVocabulary.get_word_type('<+>')
        assert actual == WordType.Lower, actual

    def test_pad_text(self):
        tk = WordVocabulary()
        value_list = [tk.vocab.SOS_index] + tk.encode_word('myNameNBA') + [tk.vocab.EOS_index]
        result = tk.decode_value_list(value_list)
        assert result == 'myNameNBA'


if __name__ == "__main__":
    unittest.main()
