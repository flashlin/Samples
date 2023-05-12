import unittest

def create_mask_text(text, offset, mask_len=1):
    prev_text = text[: offset]
    curr_text = text[offset : offset + mask_len]
    after_text = text[offset + mask_len:]
    return prev_text + "<mask>" + after_text + "\0" + curr_text

def create_mask_texts(text, mask_len=1):
    mask_texts = []
    offset = 0
    for idx in range(offset, len(text)-mask_len):
        mask_text = create_mask_text(text, idx, mask_len)
        mask_texts.append(mask_text)
    return mask_texts


class TestMaskText(unittest.TestCase):
    def test_mask_text(self):
        actual = create_mask_text("flash", 0, 1)
        assert "<mask>lash\0f" == actual

    def test_mask_texts_1(self):
        actual = create_mask_texts("flash")
        print(f"{actual=}")
        assert ["<mask>lash\0f", "f<mask>ash\0l", "fl<mask>sh\0a", "fla<mask>h\0s"] == actual