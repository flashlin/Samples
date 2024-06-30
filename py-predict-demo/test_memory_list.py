import unittest
from utils import MemoryDict

class TestMemoryDict(unittest.TestCase):

    def setUp(self):
        self.memory = MemoryDict(max_size=3)

    def test_add_and_get(self):
        self.memory.add("Hello", "World")
        self.assertEqual(self.memory.get_next_words("Hello"), ["World"])

    def test_max_size(self):
        self.memory.add("A", "1")
        self.memory.add("B", "2")
        self.memory.add("C", "3")
        self.memory.add("D", "4")
        self.assertEqual(len(self.memory), 3)
        self.assertIsNone(self.memory.get_next_word("A"))
        self.assertIsNotNone(self.memory.get_next_word("D"))

    def test_update_existing(self):
        self.memory.add("Hello", "World")
        self.memory.add("Hello", "Python")
        self.assertEqual(self.memory.get_next_words("Hello"), ["Python", "World"])

    def test_order(self):
        self.memory.add("A", "1")
        self.memory.add("B", "2")
        self.memory.add("C", "3")
        items = self.memory.get_all_items()
        self.assertEqual(items[0], ("C", ["3"]))
        self.assertEqual(items[-1], ("A", ["1"]))

    def test_str_representation(self):
        self.memory.add("A", "1")
        self.memory.add("B", "2")
        self.assertEqual(str(self.memory), "['B', 'A']")

if __name__ == '__main__':
    unittest.main()