from functools import reduce
from typing import Generic, TypeVar


class Token:
    Undefined = 'undefined'
    String = 'string'
    Number = 'number'
    Identifier = 'identifier'
    Spaces = 'spaces'
    Operator = 'operator'
    Symbol = 'symbol'
    Keyword = 'keyword'
    Empty = '(empty)'

    def __init__(self, token_type: str, text: str, offset: int, line: int, col: int):
        self.type = token_type
        self.text = text
        self.line = line
        self.col = col
        self.offset = offset
        self.value = ''

    def __repr__(self):
        text = "(none)" if self.text is None else self.text
        text = text.replace('\n', '\\n')
        return '<"%s" %s(line %d, col %d)>' % (text, self.type, self.line, self.col)

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
        while count < length:
            node = Token(Token.Undefined, " ", self.idx + count, self.line, self.col + count)
            buff.append(node)
            count += 1
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
    token.value = token.text
    return token


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
