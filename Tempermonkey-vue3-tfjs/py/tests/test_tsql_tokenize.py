from itertools import groupby

from utils.tokenizr import tsql_tokenize, Token, TSQL_Operators_Lengths, TSQL_Operators

def test_operator_lengths():
    assert TSQL_Operators_Lengths == [2, 1]

def test_number():
    stream = "123"
    tokens = tsql_tokenize(stream)
    assert tokens[0] == Token(Token.Number, "123", 0, 0, 0)

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

if __name__ == "__main__":
    test_string()