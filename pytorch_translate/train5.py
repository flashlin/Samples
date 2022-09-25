import shlex
from itertools import chain
from csharp_tokenizer import tokenize


def text_to_words_iter(text: str):
    for word in shlex.split(text):
        print(f"'{word}'")
        new_words = tokenize(word)
        for new_word in new_words:
            yield new_word
        # yield word


def text_to_words_iter(text: str):
    for word in tokenize(text):
        yield word


def text_to_words(text: str):
    # return chain([], text_to_words_iter(text))
    return [word for word in text_to_words_iter(text)]


text = 'from tb1 in customer where id==1 && name.contains("123") select tb1.id, tb2.name'
words = text_to_words(text)

print(f"{words}")
