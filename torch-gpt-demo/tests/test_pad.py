import unittest

from translate_file_datasets import pad_words


class TestPad(unittest.TestCase):
    def test_pad(self):
        words = pad_words(['1', '2'], max_len=3, pad='9')
        assert words == ['1', '2', '9']


if __name__ == "__main__":
    unittest.main()
