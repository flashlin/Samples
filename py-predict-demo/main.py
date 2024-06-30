import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import random

class MemoryPredictor(nn.Module):
    def __init__(self, vocab_size, embed_size, nhead, num_layers, memory_size=10*1024*1024):
        super(MemoryPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_size, vocab_size)
        self.memory = OrderedDict()
        self.memory_size = memory_size
        self.current_size = 0
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    def add_to_memory(self, text, next_word):
        key = (text, next_word)
        if key in self.memory:
            del self.memory[key]
        
        while self.current_size + len(str(key)) > self.memory_size and self.memory:
            _, size = self.memory.popitem(last=False)
            self.current_size -= size
        
        self.memory[key] = len(str(key))
        self.current_size += self.memory[key]
        self.memory.move_to_end(key)
    
    def train_on_memory(self, batch_size=32, epochs=1):
        if len(self.memory) < batch_size:
            print("Not enough data in memory to train.")
            return
        
        for epoch in range(epochs):
            total_loss = 0
            batches = 0
            
            memory_items = list(self.memory.keys())
            random.shuffle(memory_items)
            
            for i in range(0, len(memory_items), batch_size):
                batch = memory_items[i:i+batch_size]
                
                input_texts = [item[0] for item in batch]
                next_words = [item[1] for item in batch]
                
                input_indices = [self.text_to_index(text) for text in input_texts]
                target_indices = [self.word_to_index(word) for word in next_words]
                
                input_tensor = torch.tensor(input_indices)
                target_tensor = torch.tensor(target_indices)
                
                self.optimizer.zero_grad()
                
                output = self(input_tensor)
                loss = self.criterion(output, target_tensor)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batches += 1
            
            avg_loss = total_loss / batches
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add sequence length dimension
        output = self.transformer(x)
        return self.fc(output[-1])
    
    def predict(self, text):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            input_indices = self.text_to_index(text)
            input_tensor = torch.tensor([input_indices])
            
            output = self(input_tensor)
            probs = torch.softmax(output, dim=-1)
            
            _, top_indices = torch.topk(probs[0], 3)
            top_3 = [self.index_to_word(idx.item()) for idx in top_indices]
        
        self.train()  # Set the model back to training mode
        return top_3
    
    def text_to_index(self, text):
        # 這個方法應該被實現來將文本轉換為token索引
        pass
    
    def word_to_index(self, word):
        # 這個方法應該被實現來將單詞轉換為其索引
        pass
    
    def index_to_word(self, index):
        # 這個方法應該被實現來將索引轉換回單詞
        pass