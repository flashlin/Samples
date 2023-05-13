import unittest
from text_module import create_mask_text, create_mask_texts


class TestMaskText(unittest.TestCase):
    def test_mask_text(self):
        actual = create_mask_text("flash", 0, 1)
        assert None is actual

    def test_mask_texts_1(self):
        actual = create_mask_texts("select flash")
        assert ['<mask> flash<eos>select'] == actual, actual

    def test_mask_texts_2(self):
        actual = create_mask_texts("select id,name from customer", 1)
        assert ['<mask> id,name from customer<eos>select',
                'select id,<mask> from customer<eos>name',
                'select id,name <mask> customer<eos>from'] == actual

    def test_mask_texts_3(self):
        actual = create_mask_texts("select id,name from customer", 3)
        assert ['select <mask> from customer<eos>id,name', 'select id,<mask> customer<eos>name from'] == actual

    def test_mask_texts_4(self):
        actual = create_mask_texts("flash", 4)
        assert [] == actual
