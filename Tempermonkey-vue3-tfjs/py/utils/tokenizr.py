from functools import reduce
from typing import TypeVar, Generic

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
        text = "None" if self.text is None else self.text
        return '<%s (offset %d, line %d, col %d)>' % (text.replace('\n', '\\n'), self.offset, self.line, self.col)

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

def read_number(node: Token, stream_iterator: StreamIterator):
    buff = [ node ]
    while not stream_iterator.is_done():
        next_node = stream_iterator.next()
        if next_node.text.isdigit():
            buff.append(next_node)
        else:
            stream_iterator.prev()
            break
    return reduce_token_list(Token.Number, buff)

def read_single_quote_string(node: Token, stream_iterator: StreamIterator):
    buff = [node]
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


def tsql_tokenize(stream) -> list[Token]:
    tokens = []
    stream_iterator = StreamIterator(stream)
    c = 0
    while not stream_iterator.is_done():
        c += 1
        node = stream_iterator.next()
        if node.text.isdigit():
            tokens.append(read_number(node, stream_iterator))
            continue
        if node.text == "'":
            tokens.append(read_single_quote_string(node, stream_iterator))
            continue
        raise Exception(f"Parse token fail, unknown '{node=}' at {stream_iterator.idx=}")
    return tokens
