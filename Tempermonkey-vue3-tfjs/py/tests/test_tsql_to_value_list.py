from utils.tokenizr import tokens_to_index_list, index_list_to_string
from utils.tsql_tokenizr import tsql_tokenize, tsql_char2index_dict, tsql_index2char_dict, tsql_encode, tsql_decode


def test_number():
    tokens = tsql_tokenize("123")
    values = tokens_to_index_list(tsql_char2index_dict, tokens)
    assert values == [1, 103, 56, 57, 58, 0, 2]

def test_select_name():
    tokens = tsql_tokenize("select name")
    values = tokens_to_index_list(tsql_char2index_dict, tokens)
    assert values == [1, 102, 205, 105, 97, 101, 16, 3, 15, 7, 0, 2]

def test_tsql_values_to_string():
    values = tsql_encode("select name")
    text = index_list_to_string(tsql_index2char_dict, values)
    assert text == "SELECT name"

def test_tsql_number():
    sql = "123"
    values = tsql_encode(sql)
    text = tsql_decode(values)
    assert text == sql