from utils.tokenizr import Token, ReservedWords, index_of
from utils.tsql_tokenizr import tsql_tokenize, TSQL_Operators_Lengths, TSQL_Keywords

letters = [ch for ch in
           "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`1234567890-=~!@#$%^&*()_+{}|[]\\:\";'<>?,./ "]

tsql_marks = letters + ReservedWords + [
    "",
    "<begin>",
    "<end>",
] + TSQL_Keywords

def list_to_dict(str_list: list[str]):
    dictionary = {}
    for idx, key in enumerate(str_list):
        dictionary[key] = idx
    return dictionary

tsql_token_to_value = list_to_dict(tsql_marks)


def token_to_index(str_to_int_dict, token: Token):
    if index_of([Token.Identifier, Token.Number], token.type) != -1:
        values = [str_to_int_dict[token.type]]
        for ch in [ch for ch in token.value]:
            values.append(str_to_int_dict[ch])
        return values
    return str_to_int_dict[token.value]

def tokens_to_index(str_to_int_dict, token_list: list[Token]):
    values = [str_to_int_dict['<begin>']]
    for token in token_list:
        values += token_to_index(str_to_int_dict, token)
    values.append(str_to_int_dict['<end>'])
    return values

def test_number():
    tokens = tsql_tokenize("123")
    values = tokens_to_index(tsql_token_to_value, tokens)
    assert values == [103, 97, 53, 54, 55, 104]
    pass
