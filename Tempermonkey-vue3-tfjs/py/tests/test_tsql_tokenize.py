from utils.tokenizr import tsql_tokenize, CharNode


def test_number():
    stream = "123"
    tokens = tsql_tokenize(stream)
    assert tokens[0] == CharNode("123", 0, 0)
