from functools import reduce
from typing import TypeVar, Generic, Final
from itertools import groupby

class Token:
    Undefined = 'undefined'
    String = 'string'
    Number = 'number'
    Identifier = 'identifier'
    Operator = 'operator'
    Symbol = 'symbol'
    Empty = '(empty)'

    def __init__(self, token_type: str, text: str, offset: int, line: int, col: int):
        self.type = token_type
        self.text = text
        self.line = line
        self.col = col
        self.offset = offset

    def __repr__(self):
        text = "(none)" if self.text is None else self.text
        return '<"%s" (line %d, col %d)>' % (text.replace('\n', '\\n'), self.line, self.col)

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
            self.idx += 1
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
        if acc == EmptyToken:
            return item
        acc.text += item.text
        return acc
    token = reduce(reduce_token_list_fn, buff, EmptyToken)
    token.type = token_type
    return token

def read_number(stream_iterator: StreamIterator) -> Token:
    buff = []
    while not stream_iterator.is_done():
        token = stream_iterator.peek()
        if not token.text.isdigit():
            break
        buff.append(token)
        stream_iterator.next()
    if len(buff) == 0:
        return EmptyToken
    return reduce_token_list(Token.Number, buff)

def read_float_number(stream_iterator: StreamIterator) -> Token:
    integer = read_number(stream_iterator)
    text = stream_iterator.peek_str(1)
    if text != '.':
        return integer
    dot = stream_iterator.next()
    scale = read_number(stream_iterator)
    if scale == EmptyToken:
        stream_iterator.prev()
        return EmptyToken
    return reduce_token_list(Token.Number, [integer, dot, scale])

def read_identifier(stream_iterator: StreamIterator) -> Token:
    text = stream_iterator.peek_str(1)
    if not text.isalpha() and text != '_':
        return EmptyToken
    buff = [stream_iterator.next()]
    while not stream_iterator.is_done():
        token = stream_iterator.peek()
        if not token.text.isdigit() and not token.text.isalpha() and token.text != '_':
            break
        stream_iterator.next()
        buff.append(token)
    return reduce_token_list(Token.Identifier, buff)


def read_single_quote_string(stream_iterator: StreamIterator):
    if stream_iterator.peek_str(1) != "'":
        return EmptyToken
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
    return reduce_token_list(Token.String, buff)

def sort_desc(arr: list[str]) -> list[str]:
    arr.sort(key=lambda x: len(x))
    return arr[::-1]

def group_length(arr_sorted: list[str]):
    return [k for k, g in groupby(arr_sorted, key=lambda x: len(x))]

TSQL_Operators: Final[list[str]] = sort_desc(['<>', '>=', '<=', '!=', '=', '+', '-', '*', '/', '%'])
#TSQL_Operators_Lengths = [(k, list(g)) for k, g in groupby(TSQL_Operators, key=lambda x: len(x))]
TSQL_Operators_Lengths = group_length(TSQL_Operators)

TSQL_Symbols: Final[list[str]] = ['.', '(', ')', '@', '#']

def index_of(arr: list[str], search: str) -> int:
    for idx, item in enumerate(arr):
        if item == search:
            return idx
    return -1

def peek_str_by_list_contain(stream_iterator: StreamIterator, peek_length_list: list[int], str_list: list[str]):
    hint_length = 0
    def peek_str(length: int) -> str:
        return stream_iterator.peek_str(length)
    for peek_length in peek_length_list:
        text = peek_str(peek_length)
        if index_of(str_list, text) >= 0:
            hint_length = peek_length
            break
    return hint_length

def read_token_list_by_length(stream_iterator, hint_length):
    buff = []
    for n in range(hint_length):
        buff.append(stream_iterator.next())
    return buff

def read_operator(stream_iterator: StreamIterator):
    hint_length = peek_str_by_list_contain(stream_iterator, TSQL_Operators_Lengths, TSQL_Operators)
    buff = read_token_list_by_length(stream_iterator, hint_length)
    if not hint_length > 0:
        return EmptyToken
    return reduce_token_list(Token.Operator, buff)

def read_symbol(stream_iterator: StreamIterator):
    hint_length = peek_str_by_list_contain(stream_iterator, [1], TSQL_Symbols)
    buff = read_token_list_by_length(stream_iterator, hint_length)
    if not hint_length > 0:
        return EmptyToken
    return reduce_token_list(Token.Symbol, buff)


def try_read_any(stream_iterator: StreamIterator, fn_list: list):
    for fn in fn_list:
        token = fn(stream_iterator)
        if token != EmptyToken:
            return token
    return EmptyToken

def tsql_tokenize(stream) -> list[Token]:
    tokens = []
    stream_iterator = StreamIterator(stream)

    read_fn_list = [
        read_identifier,
        read_float_number,
        read_single_quote_string,
        read_operator,
        read_symbol
    ]

    while not stream_iterator.is_done():
        token = try_read_any(stream_iterator, read_fn_list)
        if token != EmptyToken:
            tokens.append(token)
            continue
        raise Exception(f"try to tokenize fail at {stream_iterator.idx=} '{stream_iterator.peek_str(10)}'")
    return tokens
