import string

from data_utils import sort_by_len_desc
from stream_utils import StreamTokenIterator, index_of, EmptyToken, reduce_token_list, Token

LETTERS = string.ascii_letters + " " + string.digits + string.punctuation + "\r\n\0"
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'


def peek_str_by_list_contain(stream_iterator: StreamTokenIterator,
                             peek_length_list: list[int],
                             str_list: list[str],
                             case_insensitive: bool = False):
    hint_length = 0

    def peek_str(length: int) -> str:
        return stream_iterator.peek_str(length)

    value = None
    for peek_length in peek_length_list:
        text = peek_str(peek_length)
        index = index_of(str_list, text, case_insensitive)
        if index >= 0:
            value = str_list[index]
            hint_length = peek_length
            break

    return hint_length, value


def read_token_list_by_length(stream_iterator, hint_length):
    buff = []
    for n in range(hint_length):
        buff.append(stream_iterator.next())
    return buff


def read_keyword_fn(token_type: str, length_list, keyword_list, case_insensitive: bool = False):
    def fn(stream_iterator):
        hint_length, keyword_value = peek_str_by_list_contain(stream_iterator, length_list, keyword_list,
                                                              case_insensitive)
        buff = read_token_list_by_length(stream_iterator, hint_length)
        if not hint_length > 0:
            return EmptyToken
        token = reduce_token_list(token_type, buff)
        token.value = keyword_value
        return token

    return fn


def try_read_any(stream_iterator: StreamTokenIterator, fn_list: list) -> Token:
    for fn in fn_list:
        token = fn(stream_iterator)
        if token != EmptyToken:
            return token
    return EmptyToken
