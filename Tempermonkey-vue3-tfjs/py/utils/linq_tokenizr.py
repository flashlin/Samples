from typing import Final

from utils.tokenizr import try_read_any, read_keyword_fn, fixed_marks, tokens_to_index_list, index_list_to_string, LINQ_Keywords, VOCAB_MARKS
from utils.data_utils import sort_desc, group_to_lengths, create_char2index_map, create_index2char_map
from utils.stream import StreamTokenIterator, Token, EmptyToken, read_identifier_token, read_float_number_token, read_spaces_token, \
    read_single_quote_string_token, read_double_quote_string_token

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
    stream_iterator = StreamTokenIterator(stream)

    read_fn_list = [
        read_linq_keyword_fn(),
        read_identifier_token,
        read_float_number_token,
        read_double_quote_string_token,
        read_single_quote_string_token,
        read_operator_fn(),
        read_symbol_fn(),
        read_spaces_token,
    ]

    while not stream_iterator.is_done():
        token = try_read_any(stream_iterator, read_fn_list)
        if token != EmptyToken:
            tokens.append(token)
            continue
        raise Exception(f"try to tokenize linq fail at {stream_iterator.idx=} '{stream_iterator.peek_str(10)}'")
    return tokens


LINQ_Marks = fixed_marks + LINQ_Keywords
# LINQ_Marks = VOCAB_MARKS
LINQ_Char2Index_Dict = create_char2index_map(LINQ_Marks)
LINQ_Index2Char_Dict = create_index2char_map(LINQ_Marks)
LINQ_VOCAB_SIZE = len(LINQ_Marks)

def linq_encode(stream):
    tokens = linq_tokenize(stream)
    values = tokens_to_index_list(LINQ_Char2Index_Dict, tokens)
    return values
def linq_decode(values):
    return index_list_to_string(LINQ_Index2Char_Dict, values)
