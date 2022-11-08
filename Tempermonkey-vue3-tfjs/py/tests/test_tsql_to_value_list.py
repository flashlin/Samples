from utils.tokenizr import Token
from utils.tsql_tokenizr import tsql_tokenize, TSQL_Operators_Lengths, TSQL_Keywords

letters = [ch for ch in
           "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`1234567890-=~!@#$%^&*()_+{}|[]\\:\";'<>?,./ "]

tsql_marks = letters + [
    "",
    "<begin>",
    "<end>",
] + TSQL_Keywords

def list_to_dict(str_list: list[str]):
    dictionary = {}
    for idx, key in enumerate(str_list):
        dictionary[key] = idx
    return dictionary

tsql_to_value = list_to_dict(tsql_marks)

def test_number():
    tokens = tsql_tokenize("123")
    # tokens_to_index(tsql_dict, tokens)
    #assert tsql_to_value == {}
    pass
