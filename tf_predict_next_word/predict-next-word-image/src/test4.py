import tensorflow as tf
from transformers import BertTokenizer

# 載入 BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def encode_decode(text):
    print(f'{text=}')
    # 將單詞轉換為對應的 token ID
    tokens = tokenizer.encode(text)
    # 將 token ID 轉換為張量
    input_ids = tf.constant(tokens)[None, :]  # 新增一個維度作為 batch size
    print(f'{tokens=}')
    text_restored = tokenizer.decode(tokens)
    print(f'{text_restored=}')
    print('')

# 要轉換成向量的單詞
encode_decode("Id1")
encode_decode("Id 1")
encode_decode("Id Name")
encode_decode("IdName")
