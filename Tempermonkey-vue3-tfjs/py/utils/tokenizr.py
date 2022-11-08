from functools import reduce
from typing import TypeVar, Generic, Final
import numpy as np
from itertools import groupby

class Token:
    Undefined = 'undefined'
    String = 'string'
    Number = 'number'
    Operator = 'operator'
    Empty = '(empty)'

    def __init__(self, token_type: str, text: str, offset: int, line: int, col: int):
        self.type = token_type
        self.text = text
        self.line = line
        self.col = col
        self.offset = offset

    def __repr__(self):
        text = "(none)" if self.text is None else self.text
        return '<"%s" (offset %d, line %d, col %d)>' % (text.replace('\n', '\\n'), self.offset, self.line, self.col)

    def __str__(self):
        return self.text

    def __eq__(self, other):
        return self.type == other.type \
               and self.text == other.text \
               and self.offset == other.offset \
               and self.line == other.line \
               and self.col == other.col

EmptyToken = Token(Token.Empty, None, -1, -1, -1)

T = TypeVar("T")

class StreamIterator(Generic[T]):
    def __init__(self, stream: list[T]):
        self.stream = stream
        self.length = len(stream)
        self.line = 0
        self.col = 0
        self.idx = 0
        self.buffer: list[Token] = []
        self.buffer_len = 0

    def peek(self, n=1):
        node = self.next(n)
        self.prev(n)
        return node

    def peek_str(self, length: int):
        buff = []
        count = 0
        while count < length and not self.is_done():
            node = self.next()
            buff.append(node)
            count += 1
        self.prev(count)
        buff = filter(lambda n: n.text is not None, buff)
        buff = map(lambda n: n.text, buff)
        return "".join(buff)

    def prev(self, n=1):
        node = EmptyToken
        count = 0
        while count < n:
            node = self.prev_token()
            count += 1
        return node

    def prev_token(self):
        if self.idx - 1 < 0:
            return EmptyToken
        self.idx -= 1
        return self.buffer[self.idx]

    def next(self, n=1):
        node = EmptyToken
        count = 0
        while count < n:
            node = self.next_token()
            count += 1
        return node

    def next_token(self):
        def increase_idx(ch):
            self.idx += 1
            if ch == '\n':
                self.line += 1
                self.col = 0
            else:
                self.col += 1
        if self.idx >= self.length:
            return EmptyToken
        if self.idx < self.buffer_len:
            buffer_node = self.buffer[self.idx]
            increase_idx(buffer_node.text)
            return buffer_node
        character = self.stream[self.idx]
        token = Token(Token.Undefined, character, self.idx, self.line, self.col)
        self.buffer.append(token)
        self.buffer_len += 1
        increase_idx(character)
        return token

    def is_done(self):
        if self.idx == 0 and self.length == 0:
            return True
        return self.idx >= self.length


def reduce_token_list(token_type: str, buff: list[Token]):
    def reduce_token_list_fn(acc: Token, item: Token):
        if acc.text is None:
            return item
        acc.text += item.text
        return acc
    token = reduce(reduce_token_list_fn, buff, EmptyToken)
    token.type = token_type
    return token

def try_read_number(stream_iterator: StreamIterator):
    buff = []
    while not stream_iterator.is_done():
        token = stream_iterator.peek()
        if not token.text.isdigit():
            break
        buff.append(token)
        stream_iterator.next()
    success = len(buff) > 0
    token = EmptyToken if not success else reduce_token_list(Token.Number, buff)
    return success, token

def try_read_single_quote_string(stream_iterator: StreamIterator):
    if stream_iterator.peek_str(1) != "'":
        return False, EmptyToken
    buff = [stream_iterator.next()]
    while not stream_iterator.is_done():
        next_node = stream_iterator.next()
        if next_node.text == "'":
            buff.append(next_node)
            node2 = stream_iterator.peek()
            if node2.text == "'":
                buff.append(node2)
                stream_iterator.next()
                continue
            break
        buff.append(next_node)
    return True, reduce_token_list(Token.String, buff)

def sort_desc(arr: list[str]) -> list[str]:
    arr.sort(key=lambda x: len(x))
    return arr[::-1]

def group_length(arr_sorted: list[str]):
    return [k for k, g in groupby(arr_sorted, key=lambda x: len(x))]

TSQL_Operators: Final[list[int]] = sort_desc(['<>', '>=', '<=', '!=', '=', '+', '-', '*', '/', '%'])
#TSQL_Operators_Lengths = [(k, list(g)) for k, g in groupby(TSQL_Operators, key=lambda x: len(x))]
TSQL_Operators_Lengths = group_length(TSQL_Operators)

def try_read_operator(stream_iterator: StreamIterator):
    buff = []
    hint_length = 0
    def peek_str(length: int) -> str:
        return stream_iterator.peek_str(length)
    for read_len in TSQL_Operators_Lengths:
        text = peek_str(read_len)
        if TSQL_Operators.index(text) >= 0:
            hint_length = read_len
            break
    for n in range(hint_length):
        buff.append(stream_iterator.next())
    success = hint_length > 0
    token = EmptyToken if not success else reduce_token_list(Token.Operator, buff)
    return success, token

def tsql_tokenize(stream) -> list[Token]:
    tokens = []
    stream_iterator = StreamIterator(stream)
    while not stream_iterator.is_done():
        is_number, number = try_read_number(stream_iterator)
        if is_number:
            tokens.append(number)
            continue
        is_string, string = try_read_single_quote_string(stream_iterator)
        if is_string:
            tokens.append(string)
            continue
        is_operator, operator = try_read_operator(stream_iterator)
        if is_operator:
            tokens.append(operator)
            continue
        raise Exception(f"Parse token fail at {stream_iterator.idx=} '{stream_iterator.peek_str(10)}'")
    return tokens
