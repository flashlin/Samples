import unittest
from vocabulary_utils import WordVocabulary


class TestWordVocabulary(unittest.TestCase):
    def test_encode_decode(self):
        tk = WordVocabulary()
        index_list = tk.encode_many_words(['hello', ' ', 'World'])
        text = tk.decode_index_list(index_list)
        assert text == "hello World"

    def test_split_text(self):
        word_list = WordVocabulary.split_text('hello1_123_1my_Name')
        assert word_list == ['hello1', '_', '123', '_', '1', 'my', '_',  'Name'], word_list

    def test_split_text2(self):
        word_list = WordVocabulary.split_text('hello1myName')
        assert word_list == ['hello1', 'my', 'Name'], word_list

    def test_split_text3(self):
        word_list = WordVocabulary.split_text('myIdName')
        assert word_list == ['my', 'Id', 'Name'], word_list


if __name__ == "__main__":
    unittest.main()
