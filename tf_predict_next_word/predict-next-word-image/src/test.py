import unittest

from SimpleTokenizer import SimpleTokenizer


class TestAdd(unittest.TestCase):
    def test_add(self):
        tokenizer = SimpleTokenizer()
        text = '<EOS>'
        tokenizer.fit_on_texts(text)
        sequences = tokenizer.encode(text)
        print(f'{sequences=}')
        text_restored = tokenizer.decode(sequences)
        self.assertEqual(text_restored, text)


if __name__ == '__main__':
    unittest.main()