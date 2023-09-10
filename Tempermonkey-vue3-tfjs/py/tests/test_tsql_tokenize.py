from utils.stream import Token
from utils.tsql_tokenizr import tsql_tokenize, TSQL_Operators_Lengths


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

def test_string1():
    stream = "'123','abc'"
    tokens = tsql_tokenize(stream)
    assert tokens[0] == Token(Token.String, "'123'", 0, 0, 0)
    assert tokens[1] == Token(Token.Symbol, ",", 5, 0, 5)
    assert tokens[2] == Token(Token.String, "'abc'", 6, 0, 6)

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

def test_select_from_table():
    tokens = tsql_tokenize("select tb1.name from customer")
    assert tokens == [
        Token(Token.Keyword, "select", 0, 0, 0),
        Token(Token.Spaces, " ", 6, 0, 6),
        Token(Token.Identifier, "tb1", 7, 0, 7),
        Token(Token.Symbol, ".", 10, 0, 10),
        Token(Token.Identifier, "name", 11, 0, 11),
        Token(Token.Spaces, " ", 15, 0, 15),
        Token(Token.Keyword, "from", 16, 0, 16),
        Token(Token.Spaces, " ", 20, 0, 20),
        Token(Token.Identifier, "customer", 21, 0, 21),
    ]

if __name__ == "__main__":
    test_string()