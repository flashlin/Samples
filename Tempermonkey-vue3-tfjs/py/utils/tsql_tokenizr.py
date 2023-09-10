from typing import Final

from utils.tokenizr import try_read_any, read_keyword_fn, fixed_marks, tokens_to_index_list, index_list_to_string, TSQL_Keywords, VOCAB_MARKS
from utils.data_utils import sort_desc, group_to_lengths, create_char2index_map, create_index2char_map
from utils.stream import StreamTokenIterator, Token, EmptyToken, reduce_token_list, read_identifier_token, read_float_number_token, \
    read_spaces_token, read_single_quote_string_token

TSQL_Keywords_Lengths = group_to_lengths(TSQL_Keywords)
TSQL_Symbols: Final[list[str]] = [',', '.', '(', ')', '@', '#']
TSQL_Operators: Final[list[str]] = sort_desc(['<>', '>=', '<=', '!=', '=', '+', '-', '*', '/', '%'])
# TSQL_Operators_Lengths = [(k, list(g)) for k, g in groupby(TSQL_Operators, key=lambda x: len(x))]
TSQL_Operators_Lengths = group_to_lengths(TSQL_Operators)


def read_tsql_keyword_fn():
    return read_keyword_fn(Token.Keyword, TSQL_Keywords_Lengths, TSQL_Keywords, case_insensitive=True)


def read_symbol_fn():
    return read_keyword_fn(Token.Symbol, [1], TSQL_Symbols)


def read_operator_fn():
    return read_keyword_fn(Token.Operator, TSQL_Operators_Lengths, TSQL_Operators)


def read_tsql_identifier(stream_iterator):
    if stream_iterator.peek_str(1) != "[":
        return EmptyToken
    buff = [stream_iterator.next()]
    while not stream_iterator.is_done():
        token = stream_iterator.peek()
        if token.text == "]":
            stream_iterator.next()
            buff.append(token)
            break
        stream_iterator.next()
        buff.append(token)
    return reduce_token_list(Token.Identifier, buff)


def tsql_tokenize(stream) -> list[Token]:
    tokens = []
    stream_iterator = StreamTokenIterator(stream)

    read_fn_list = [
        read_tsql_keyword_fn(),
        read_identifier_token,
        read_float_number_token,
        read_single_quote_string_token,
        read_operator_fn(),
        read_symbol_fn(),
        read_spaces_token,
        read_tsql_identifier
    ]

    while not stream_iterator.is_done():
        token = try_read_any(stream_iterator, read_fn_list)
        if token != EmptyToken:
            tokens.append(token)
            continue
        raise Exception(f"try to tokenize fail at {stream_iterator.idx=} '{stream_iterator.peek_str(10)}'")
    return tokens


tsql_marks = fixed_marks + TSQL_Keywords
# tsql_marks = VOCAB_MARKS
TSQL_CHAR2INDEX_DICT = create_char2index_map(tsql_marks)
TSQL_INDEX2CHAR_DICT = create_index2char_map(tsql_marks)
TSQL_VOCAB_SIZE = len(tsql_marks)


def tsql_encode(stream):
    tokens = tsql_tokenize(stream)
    values = tokens_to_index_list(TSQL_CHAR2INDEX_DICT, tokens)
    return values


def tsql_decode(values):
    return index_list_to_string(TSQL_INDEX2CHAR_DICT, values)
