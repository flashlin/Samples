from utils.tokenizr import ReservedWords, tokens_to_index, str_list_to_dict
from utils.tsql_tokenizr import tsql_tokenize, TSQL_Keywords

letters = [ch for ch in
           "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`1234567890-=~!@#$%^&*()_+{}|[]\\:\";'<>?,./ "]

tsql_marks = letters + ReservedWords + [
    "",
    "<begin>",
    "<end>",
] + TSQL_Keywords

tsql_token_to_value = str_list_to_dict(tsql_marks)

def test_number():
    tokens = tsql_tokenize("123")
    values = tokens_to_index(tsql_token_to_value, tokens)
    assert values == [103, 97, 53, 54, 55, 104]
    pass
