from itertools import groupby

from utils.tokenizr import tsql_tokenize, Token, TSQL_Operators_Lengths

def test_operator_lengths():
    assert TSQL_Operators_Lengths == [2, 1]

def test_number():
    stream = "123"
    tokens = tsql_tokenize(stream)
    assert tokens[0] == Token(Token.Number, "123", 0, 0, 0)

def test_float():
    stream = "123.33"
    tokens = tsql_tokenize(stream)
    assert tokens[0] == Token(Token.Number, stream, 0, 0, 0)

def test_string():
    stream = "'abc ''123'''"
    tokens = tsql_tokenize(stream)
    assert tokens[0] == Token(Token.String, stream, 0, 0, 0)


def test_operator():
    stream = "<>"
    tokens = tsql_tokenize(stream)
    assert tokens[0] == Token(Token.Operator, stream, 0, 0, 0)

def test_symbol():
    stream = "."
    tokens = tsql_tokenize(stream)
    assert tokens[0] == Token(Token.Symbol, stream, 0, 0, 0)

def test_tb1_name():
    tokens = tsql_tokenize("tb1.name")
    assert tokens == [
        Token(Token.Identifier, "tb1", 0, 0, 0),
        Token(Token.Symbol, ".", 3, 0, 3),
        Token(Token.Identifier, "name", 4, 0, 4)
    ]

def test_select_keyword():
    tokens = tsql_tokenize("select")
    assert tokens == [
        Token(Token.Keyword, "select", 0, 0, 0)
    ]

if __name__ == "__main__":
    test_string()