from utils.tokenizr import tokens_to_index_list, index_list_to_string
from utils.tsql_tokenizr import tsql_tokenize, TSQL_CHAR2INDEX_DICT, TSQL_INDEX2CHAR_DICT, tsql_encode, tsql_decode


def test_number():
    tokens = tsql_tokenize("123")
    values = tokens_to_index_list(TSQL_CHAR2INDEX_DICT, tokens)
    assert values == [1, 104, 57, 58, 59, 0, 2]

def test_select_name():
    tokens = tsql_tokenize("select name")
    values = tokens_to_index_list(TSQL_CHAR2INDEX_DICT, tokens)
    assert values == [1, 103, 206, 106, 98, 102, 17, 4, 16, 8, 0, 2]

def test_tsql_values_to_string():
    values = tsql_encode("select name")
    text = tsql_decode(values)
    assert text == "SELECT name"

def test_tsql_number():
    sql = "123"
    values = tsql_encode(sql)
    text = tsql_decode(values)
    assert text == sql