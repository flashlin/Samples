import torch
from collections import OrderedDict


class MemoryDict:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.memory = OrderedDict()

    def add(self, key: str, next_word: str):
        element = None
        if key in self.memory:
            element = self.memory.pop(key)
        else:
            element = MemoryDict(3)

        if len(self.memory) >= self.max_size:
            self.memory.popitem(last=True)

        element.internal_add(next_word, next_word)
        self.memory[key] = element
        self.memory.move_to_end(key, last=False)

    def internal_add(self, key: str, next_word: str):
        if key in self.memory:
            self.memory.pop(key)
        if len(self.memory) >= self.max_size:
            self.memory.popitem(last=True)
        self.memory[key] = next_word
        self.memory.move_to_end(key, last=False)    
    
    def get_next_words(self, key: str):
        if key in self.memory:
            return self.memory[key].get_all_values()
        return []

    def get_next_word(self, key: str):
        if key in self.memory:
            return self.memory[key].get_all_values()[0]
        return None

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        return str(list(self.memory.keys()))

    def get_all_items(self):
        for k, v in list(self.memory.items()):
            sub_values = v.get_all_values()
            for value in sub_values:
                yield k, value
    
    def get_all_values(self):
        return list(self.memory.values())
    