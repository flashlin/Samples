import unittest
from vocabulary_utils import WordVocabulary


class TestWordVocabulary(unittest.TestCase):
    def test1(self):
        tk = WordVocabulary()
        index_list = tk.encode_many_words(['hello', ' ', 'World'])
        text = tk.decode_index_list(index_list)
        assert text == "hello World"


if __name__ == "__main__":
    unittest.main()
