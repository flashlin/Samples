import spacy
model = spacy.load("en_core_web_sm")
print(model._path)