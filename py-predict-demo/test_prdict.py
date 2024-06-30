import unittest
import torch
from main import MemoryPredictor

class TestMemoryPredictor(unittest.TestCase):

    def setUp(self):
        self.model = MemoryPredictor()

    def test_text_to_index(self):
        text = "Hello"
        indices = self.model.text_to_index(text)
        self.assertEqual(indices, [72, 101, 108, 108, 111])

    def test_word_to_index(self):
        word = "A"
        index = self.model.word_to_index(word)
        self.assertEqual(index, 65)

    def test_index_to_word(self):
        index = 65
        word = self.model.index_to_word(index)
        self.assertEqual(word, "A")

    def test_add_to_memory(self):
        self.model.add_to_memory("Hello", "W")
        memory_items = self.model.memory.get_all_items()
        self.assertIn(("Hello", "W"), memory_items)

    def test_predict(self):
        # 這個測試僅檢查 predict 方法是否返回預期的數據類型和長度
        prediction = self.model.predict("Hello")
        self.assertIsInstance(prediction, list)
        self.assertEqual(len(prediction), 3)
        for word in prediction:
            self.assertIsInstance(word, str)
            self.assertEqual(len(word), 1)

    def test_save_and_load_weights(self):
        self.model.save_weights('outputs/test_weights.pt')
        # 創建一個新模型並載入權重
        new_model = MemoryPredictor()
        new_model.load_weights('outputs/test_weights.pt')
        
        # 檢查兩個模型的狀態字典是否相同
        for (k1, v1), (k2, v2) in zip(self.model.state_dict().items(), new_model.state_dict().items()):
            self.assertEqual(k1, k2)
            self.assertTrue(torch.equal(v1, v2))

if __name__ == '__main__':
    unittest.main()