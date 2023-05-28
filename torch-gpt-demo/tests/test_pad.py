import unittest

from translate_file_datasets import pad_words, pad_zip_words


class TestPad(unittest.TestCase):
    def test_pad(self):
        words = pad_words(['1', '2'], max_len=3, pad='9')
        assert words == ['1', '2', '9']

    def test_pad_zip(self):
        a = ['a', 'b']
        b = ['1', '2', '3', '4']
        words = pad_zip_words(a, b, max_len=2, pad='P')
        print(f'{words=}')
        assert words == [
            [['a', 'b'], ['1', '2']],
            [['b', 'P'], ['2', '3']],
            [['P', 'P'], ['3', '4']],
        ]


if __name__ == "__main__":
    unittest.main()
