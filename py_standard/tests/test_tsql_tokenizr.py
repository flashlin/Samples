import unittest
from stream_utils import Token, NUMBER, STRING, SYMBOL, OPERATOR, IDENTIFIER, KEYWORD, SPACES
from tsql_tokenizr import TSQL_OPERATORS_LENGTHS, tsql_tokenize


class TestTsqlTokenizer(unittest.TestCase):
    def test_operator_lengths(self):
        assert TSQL_OPERATORS_LENGTHS == [2, 1]

    def test_number(self):
        stream = "123"
        tokens = tsql_tokenize(stream)
        assert tokens[0] == Token(NUMBER, "123", 0, 0, 0)

    def test_float(self):
        stream = "123.33"
        tokens = tsql_tokenize(stream)
        assert tokens[0] == Token(NUMBER, stream, 0, 0, 0)

    def test_string(self):
        stream = "'abc ''123'''"
        tokens = tsql_tokenize(stream)
        assert tokens[0] == Token(STRING, stream, 0, 0, 0)

    def test_string1(self):
        stream = "'123','abc'"
        tokens = tsql_tokenize(stream)
        assert tokens[0] == Token(STRING, "'123'", 0, 0, 0)
        assert tokens[1] == Token(SYMBOL, ",", 5, 0, 5)
        assert tokens[2] == Token(STRING, "'abc'", 6, 0, 6)

    def test_operator(self):
        stream = "<>"
        tokens = tsql_tokenize(stream)
        assert tokens[0] == Token(OPERATOR, stream, 0, 0, 0)

    def test_symbol(self):
        stream = "."
        tokens = tsql_tokenize(stream)
        assert tokens[0] == Token(SYMBOL, stream, 0, 0, 0)

    def test_tb1_name(self):
        tokens = tsql_tokenize("tb1.name")
        assert tokens == [
            Token(IDENTIFIER, "tb1", 0, 0, 0),
            Token(SYMBOL, ".", 3, 0, 3),
            Token(IDENTIFIER, "name", 4, 0, 4)
        ]

    def test_select_keyword(self):
        tokens = tsql_tokenize("select")
        assert tokens == [
            Token(KEYWORD, "select", 0, 0, 0)
        ]

    def test_select_from_table(self):
        tokens = tsql_tokenize("select tb1.name from customer")
        assert tokens == [
            Token(KEYWORD, "select", 0, 0, 0),
            Token(SPACES, " ", 6, 0, 6),
            Token(IDENTIFIER, "tb1", 7, 0, 7),
            Token(SYMBOL, ".", 10, 0, 10),
            Token(IDENTIFIER, "name", 11, 0, 11),
            Token(SPACES, " ", 15, 0, 15),
            Token(KEYWORD, "from", 16, 0, 16),
            Token(SPACES, " ", 20, 0, 20),
            Token(IDENTIFIER, "customer", 21, 0, 21),
        ]


if __name__ == "__main__":
    unittest.main()
