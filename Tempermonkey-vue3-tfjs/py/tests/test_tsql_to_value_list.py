from utils.tokenizr import tokens_to_index
from utils.tsql_tokenizr import tsql_tokenize, tsql_token_to_value


def test_number():
    tokens = tsql_tokenize("123")
    values = tokens_to_index(tsql_token_to_value, tokens)
    assert values == [103, 97, 53, 54, 55, 104]

def test_select_name():
    tokens = tsql_tokenize("select name")
    print(f"{tsql_token_to_value['spaces']=}")
    values = tokens_to_index(tsql_token_to_value, tokens)
    assert values == [103, 202, 94, 95, 13, 0, 12, 4, 104]
