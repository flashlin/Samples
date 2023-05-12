import unittest
from text_module import create_mask_text, create_mask_texts


class TestMaskText(unittest.TestCase):
    def test_mask_text(self):
        actual = create_mask_text("flash", 0, 1)
        assert "<mask>lash\0f" == actual

    def test_mask_texts_1(self):
        actual = create_mask_texts("flash")
        assert ["<mask>lash\0f", "f<mask>ash\0l", "fl<mask>sh\0a", "fla<mask>h\0s"] == actual

    def test_mask_texts_2(self):
        actual = create_mask_texts("flash", 2)
        assert ["<mask>ash\0fl", "f<mask>sh\0la", "fl<mask>h\0as", "fla<mask>\0sh"] == actual
