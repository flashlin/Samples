import torch
import torch.nn as nn
import torch.optim as optim
import re
from utils import MemoryDict

class MemoryPredictor(nn.Module):
    def __init__(self, vocab_size=256, embed_size=1024, nhead=4, num_layers=8):
        super(MemoryPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_size, vocab_size)
        self.memory = MemoryDict(100)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    def add_to_memory(self, text, next_word):
        self.memory.add(text, next_word)
    
    def train_on_memory(self, batch_size=32, epochs=1):
        memory_items = self.memory.get_all_items()
        
        if len(memory_items) < batch_size:
            print("Not enough data in memory to train.")
            return

        for epoch in range(epochs):
            total_loss = 0
            batches = 0
            
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
        self.save_weights('outputs/memory.pt')
        
    def save_weights(self, filename):
        torch.save(self.state_dict(), filename)
        print(f"Model weights saved to {filename}")

    def load_weights(self, filename):
        self.load_state_dict(torch.load(filename))
    
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
            
            # 改變輸入張量的形狀
            input_tensor = input_tensor.transpose(0, 1)  # 將形狀從 (1, seq_len) 變為 (seq_len, 1)
            
            x = self.embedding(input_tensor)
            # 不需要再使用 unsqueeze，因為現在已經是正確的形狀了
            output = self.transformer(x)
            output = self.fc(output[-1])  # 只取最後一個時間步的輸出
            probs = torch.softmax(output, dim=-1)
            
            _, top_indices = torch.topk(probs[0], 3)
            top_3 = [self.index_to_word(idx.item()) for idx in top_indices]
        
        self.train()
        return top_3
    
    def text_to_index(self, text):
        return [ord(char) for char in text]
    
    def word_to_index(self, word):
        return ord(word)
    
    def index_to_word(self, index):
        return chr(index)
    
    def predict_word(self, text):
        def generate_word(start_text, next_char):
            buffer = '' + next_char
            current_text = start_text + next_char
            while True:
                char = self.predict(current_text)[0]
                if char == ' ' or ord(char) == 0:
                    return buffer
                current_text += char
                buffer += char
                if len(buffer) > 45:  # 防止無限循環
                    return buffer

        top_3_words = []
        for char in self.predict(text):
            word = generate_word(text, char)
            top_3_words.append(word)
        return top_3_words
        

def split_text_to_words_list(text):
    # 使用正則表達式分割文本
    pattern = r'(\s+|[^\s\u4e00-\u9fff]+|[\u4e00-\u9fff])'
    return re.findall(pattern, text)


def words_list_to_nextword_pairwise(arr):
    result = arr[0]
    for i in range(len(arr) - 1):
        yield (result, arr[i+1])
        result += arr[i+1]

        
if __name__ == '__main__':
    # 測試函數
    test_text = "hello world 你好"
    result = split_text_to_words_list(test_text)
    for (text, next_word) in words_list_to_nextword_pairwise(result):
        print(f"{text=} {next_word=}")


    model = MemoryPredictor()
    words = model.predict_word("Hello")
    for word in words:
        print(f"{word=}")