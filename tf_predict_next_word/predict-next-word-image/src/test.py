import unittest

from MyBertTokenizer import MyBertTokenizer
from SimpleTokenizer import SimpleTokenizer


class TestAdd(unittest.TestCase):

    def test_eos(self):
        self.execute('<eos>' + chr(0))

    def test_e(self):
        self.execute(chr(0))

    def execute(self, text):
        tokenizer = MyBertTokenizer()
        # tokenizer.keep_special_tokens([text])
        sequences = tokenizer.encode(text)
        print(f'{text=} {sequences=}')
        text_restored = tokenizer.decode(sequences)
        self.assertEqual(text_restored, text)


if __name__ == '__main__':
    unittest.main()