import spacy

# Load an English word embedding model
# nlp = spacy.load("xx_sent_ud_sm")
nlp = spacy.load("en_core_web_trf")
print(nlp.path)

# 讀取本文
text = """
The quick brown fox jumps over the lazy dog.
The 狐狸 is 3 years old.
The dog is 5 years old.
"""

# 對本文進行分詞和詞性標注
doc = nlp(text)

# 提取關鍵字和數值
keywords = [token.text for token in doc if token.pos_ == "NOUN"]
numbers = [token.text for token in doc if token.pos_ == "NUM"]

# 將關鍵字和數值轉換為 metadata
metadata = {"keywords": keywords, "numbers": numbers}

metadata2 = {keyword: number for keyword, number in zip(keywords, numbers)}
#metadata2 = list(metadata2.items())

print(f"{doc=}")
print(f"{metadata=}")
print(f"{metadata2=}")

