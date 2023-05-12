import unittest

def mask_text(text, offset, mask_len=1):
    #for idx in range(offset, len(text)-offset):
    prev_text = text[: offset]
    curr_text = text[offset: mask_len]
    after_text = text[offset + mask_len:]
    return prev_text + "<mask>" + after_text + "\0" + curr_text


class TestMaskText(unittest.TestCase):
    def test_operator_lengths(self):
        actual = mask_text("flash", 0, 1)
        assert "<mask>lash\0f" == actual