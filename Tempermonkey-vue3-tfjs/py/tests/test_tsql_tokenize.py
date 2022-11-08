from utils.tokenizr import tsql_tokenize, Token


def test_number():
    stream = "123"
    tokens = tsql_tokenize(stream)
    assert tokens[0] == Token(Token.Number, "123", 0, 0, 0)

def test_string():
    stream = "'abc ''123'''"
    tokens = tsql_tokenize(stream)
    assert tokens[0] == Token(Token.String, stream, 0, 0, 0)


if __name__ == "__main__":
    test_string()