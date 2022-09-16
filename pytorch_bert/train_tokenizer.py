from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
import os

TOKENS_TEXT_FILES_PATH = "./input/tokens_text_files"
PRETRAINED_PATH = "./pretrained"

def train():
   paths = [str(x) for x in Path(TOKENS_TEXT_FILES_PATH).glob('**/*.txt')]
   # initialize
   tokenizer = ByteLevelBPETokenizer()

   # 英文單字數量, 對母語人士來說, 單字量約15,000-20,000
   # 中文 4808 數量
   tokenizer.train(files=paths, 
      vocab_size=35_522,
      min_frequency=2,
      special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])


   os.mkdir(PRETRAINED_PATH)
   tokenizer.save_model(PRETRAINED_PATH) #他會產生 merges.txt 和 vocab.json


def test():
   from transformers import RobertaTokenizerFast
   tokenizer = RobertaTokenizerFast.from_pretrained(PRETRAINED_PATH)
   lorem_ipsum = (
       "我家門口有小河, Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor"
       "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
   )

   res = tokenizer(lorem_ipsum, max_length=100, padding='max_length', truncation=True)
   print(res)
   # 我們input_ids可以看到我們的序列標記<s>的開始由0表示，序列標記的結束<s\>由2表示，填充標記<pad>由1表示
   # res = tokenizer.encode_plus("i like  you  much", "but not him")
   #id2token = tokenizer.convert_ids_to_tokens(res['input_ids'])
   #print(id2token)
   #str = ''.join(id2token)
   #print(str)
   ss = tokenizer.decode(res['input_ids'])
   print(ss)

# train()
test()
