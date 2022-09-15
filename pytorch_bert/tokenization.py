from lib2to3.pgen2.tokenize import tokenize
from sre_parse import Tokenizer
from transformers import BertTokenizer

def from_pretrained_tokenizer():
   # folder 包含有config.json和vocab.txt這兩個檔案
   # 可從這下載 https://huggingface.co/bert-base-uncased/tree/main
   TOKENIZER_PATH = "./huggingface/bert-base-cased"
   return BertTokenizer.from_pretrained(TOKENIZER_PATH)


def create_tokenizer():
   return BertTokenizer(vocab_file="./pretrain_model/bert-base-chinese/vocab.txt")

def sample1():
   tokenizer = create_tokenizer()
   examples = ["我愛你", "你愛吃炸雞"]
   res = tokenizer.tokenize(examples[0])
   token_ids = tokenizer.convert_tokens_to_ids(res)
   id2token = tokenizer.convert_ids_to_tokens(token_ids)


   # 是否需要padding，可選如下幾個值
   # Trueor 'longest'，padding到一個batch中最長序列的長度
   # 'max_length'， padding到由max_length參數指定的長度，如果沒有指定max_length則padding到模型所能接受的最大長度
   # Falseor 'do_not_pad'， 不進行padding
   res = tokenizer(
      examples,
      padding="max_length",
      truncation=True, #是否要進行截斷
      max_length=7,
      return_tensors="pt", #返回類型，默認是list類型，可選pt返回torch 的tensor，tf返回tensorflow的tensor，npnumpy類型
      return_length=True, #是否返回編碼的序列長度，default=False
   )
   # 返回一個字典
   # { 'input_ids': tensor(...),
   #   'token_type_ids": ...
   #   'length': ...
   #   'attention_mask': ...
   # }
   print(res["input_ids"].shape)

def main():
   tokenizer = from_pretrained_tokenizer()
   examples = ["print('我愛你abc')", "console.log('aaa', '你愛吃炸雞')"]
   res = tokenizer.tokenize(examples[0])
   token_ids = tokenizer.convert_tokens_to_ids(res)
   id2token = tokenizer.convert_ids_to_tokens(token_ids)
   print(res)
   print(token_ids)
   print(id2token)


if __name__ == "__main__":
   main()
