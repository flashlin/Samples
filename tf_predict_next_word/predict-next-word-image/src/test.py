import unittest

from MyBertTokenizer import MyBertTokenizer
from SimpleTokenizer import SimpleTokenizer
from Vocabulary import Vocabulary


class TestAdd(unittest.TestCase):

    def test_eos(self):
        self.encode_token('<eos>')

    def test_n_grams(self):
        corpus = [ 'select id from customer' ]
        vocab = Vocabulary(MyBertTokenizer())
        n_grams = vocab.create_n_gram_values(corpus, 10)
        for sequence in n_grams:
            text = vocab.tokenizer.decode(sequence)
            print(f'n_grams1={text=}')
            print(f'n_grams2={sequence=}')
            print()

    def encode_token(self, token):
        expected = token
        tokenizer = MyBertTokenizer()
        sequences = tokenizer.encode([token])
        print(f'test {token=} {sequences=}')
        text_restored = tokenizer.decode(sequences)
        self.assertEqual(text_restored, expected)


if __name__ == '__main__':
    unittest.main()