from typing import Final

from utils.tokenizr import Token, StreamIterator, read_identifier, read_float_number, \
    read_single_quote_string, try_read_any, EmptyToken, sort_desc, group_to_lengths, \
    read_keyword_fn, read_spaces, convert_str_list_to_char2index_map, convert_str_list_to_index2char_map, \
    fixed_marks, tokens_to_index_list, read_double_quote_string

LINQ_Keywords = sort_desc([
    "select",
    "from",
    "into",
    "equals",
    "group",
    "by",
    "join",
    "in",
    "new"
])
LINQ_Keywords_Lengths = group_to_lengths(LINQ_Keywords)
LINQ_Symbols: Final[list[str]] = ['.', ',', '(', ')', '{', '}']
LINQ_Operators: Final[list[str]] = sort_desc(['>=', '<=', '!=', '=', '+', '-', '*', '/', '%'])
LINQ_Operators_Lengths = group_to_lengths(LINQ_Operators)

def read_linq_keyword_fn():
    return read_keyword_fn(Token.Keyword, LINQ_Keywords_Lengths, LINQ_Keywords)

def read_symbol_fn():
    return read_keyword_fn(Token.Symbol, [1], LINQ_Symbols)

def read_operator_fn():
    return read_keyword_fn(Token.Operator, LINQ_Operators_Lengths, LINQ_Operators)

def linq_tokenize(stream) -> list[Token]:
    tokens = []
    stream_iterator = StreamIterator(stream)

    read_fn_list = [
        read_linq_keyword_fn(),
        read_identifier,
        read_float_number,
        read_double_quote_string,
        read_single_quote_string,
        read_operator_fn(),
        read_symbol_fn(),
        read_spaces,
    ]

    while not stream_iterator.is_done():
        token = try_read_any(stream_iterator, read_fn_list)
        if token != EmptyToken:
            tokens.append(token)
            continue
        raise Exception(f"try to tokenize linq fail at {stream_iterator.idx=} '{stream_iterator.peek_str(10)}'")
    return tokens


linq_marks = fixed_marks + LINQ_Keywords
linq_char2index_dict = convert_str_list_to_char2index_map(linq_marks)
linq_index2char_dict = convert_str_list_to_index2char_map(linq_marks)


def linq_encode(stream):
    tokens = linq_tokenize(stream)
    print(f"{tokens=}")
    values = tokens_to_index_list(linq_char2index_dict, tokens)
    return values
