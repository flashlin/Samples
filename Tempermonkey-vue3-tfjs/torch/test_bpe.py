from ml.bpe_tokenizer import SimpleTokenizer
from utils.linq_tokenizr import linq_tokenize

if __name__ == '__main__':
    tk = SimpleTokenizer(linq_tokenize)
    s1 = 'from tb1 in customer select new{ tb1.name, lastName="flash" }'
    print(f"{s1=}")
    tokens = tk.encode(s1)
    print(f"{tokens=}")
    s2 = tk.decode(tokens)
    print(f"{s2=}")
