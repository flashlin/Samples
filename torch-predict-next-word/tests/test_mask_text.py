import unittest
from text_module import create_mask_text, create_mask_texts


class TestMaskText(unittest.TestCase):
    def test_mask_text(self):
        actual = create_mask_text("flash", 0, 1)
        assert "<mask>lash<end>f" == actual

    def test_mask_texts_1(self):
        actual = create_mask_texts("flash")
        assert ["<mask>lash<end>f", "f<mask>ash<end>l", "fl<mask>sh<end>a", "fla<mask>h<end>s"] == actual

    def test_mask_texts_2(self):
        actual = create_mask_texts("flash", 2)
        assert ["<mask>ash<end>fl", "f<mask>sh<end>la", "fl<mask>h<end>as", "fla<mask><end>sh"] == actual

    def test_mask_texts_3(self):
        actual = create_mask_texts("flash", 3)
        assert ["<mask>sh<end>fla", "f<mask>h<end>las", "fl<mask><end>ash", "fla<mask><end>sh"] == actual

    def test_mask_texts_4(self):
        actual = create_mask_texts("flash", 4)
        assert ["<mask>h<end>flas", "f<mask><end>lash", "fl<mask><end>ash", "fla<mask><end>sh"] == actual

    def test_mask_texts_5(self):
        actual = create_mask_texts("flash", 5)
        assert [] == actual
