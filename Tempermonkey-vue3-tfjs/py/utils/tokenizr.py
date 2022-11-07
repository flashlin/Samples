from functools import reduce
from typing import TypeVar, Generic


class Token:
    Name = 'name'
    String = 'string'
    Number = 'number'
    Operator = 'operator'
    Boolean = 'boolean'
    Undefined = 'undefined'
    Null = 'null'
    Regex = 'regex'
    EOF = '(end)'

    LITERALS = [String, Number, Boolean, Regex, Null, Undefined]

    def __init__(self, source: str, token_type: str, line=0, char=0):
        self.value = source
        self.type = token_type
        self.line = line
        self.char = char

    def __repr__(self):
        return '<%s: %s (line %d, char %d)>' % (self.type, self.value.replace('\n', '\\n'), self.line, self.char)

    def __str__(self):
        return self.value


class CharNode:
    def __init__(self, text: str, line: int, col: int):
        self.text = text
        self.line = line
        self.col = col

    def __repr__(self):
        return '<%s (line %d, col %d)>' % (self.text.replace('\n', '\\n'), self.line, self.col)

    def __str__(self):
        return self.text

    def __eq__(self, other):
        return self.text == other.text \
               and self.line == other.line \
               and self.col == other.col

EmptyCharNode = CharNode(None, -1, -1)

T = TypeVar("T")

class StreamIterator(Generic[T]):
    def __init__(self, stream: list[T]):
        self.stream = stream
        self.length = len(stream)

    buffer = []
    buffer_len = 0
    line = 0
    col = 0
    idx = 0

    def peek(self, n=1):
        node = self.next(n)
        self.prev(n)
        return node

    def prev(self, n=1):
        node = EmptyCharNode
        count = 0
        while count < n:
            node = self.prev_node()
            count += 1
        return node

    def prev_node(self):
        if self.idx - 1 < 0:
            return EmptyCharNode
        self.idx -= 1
        return self.buffer[self.idx]

    def next(self, n=1):
        node = None
        count = 0
        while count < n:
            node = self.next_node()
            count += 1
        return node

    def next_node(self):
        if self.idx >= self.length:
            return EmptyCharNode
        if self.idx < self.buffer_len:
            return self.buffer[self.idx]
        character = self.stream[self.idx]
        node = CharNode(character, self.line, self.col)
        self.buffer.append(node)
        self.buffer_len += 1
        self.idx += 1
        if character == '\n':
            self.line += 1
            self.col = 0
        else:
            self.col += 1
        return node

    def is_done(self):
        if self.idx == 0 and self.length == 0:
            return True
        return self.idx >= self.length


def read_number(node, stream_iterator):
    buff = [ node ];
    while not stream_iterator.is_done():
        next_node = stream_iterator.next()
        if next_node.text.isdigit():
            buff.append(next_node)
        else:
            stream_iterator.prev()
            break
    def reduceChartNodeFn(acc: CharNode, item: CharNode):
        if acc.text is None:
            return item
        acc.text += item.text
        return acc
    return reduce(reduceChartNodeFn, buff, CharNode(None, -1, -1))

def tsql_tokenize(stream) -> list[Token]:
    tokens = []
    stream_iterator = StreamIterator(stream)
    while not stream_iterator.is_done():
        node = stream_iterator.next()
        if node.text.isdigit():
            token = read_number(node, stream_iterator)
            tokens.append(token)
            continue
        raise Exception(f"Parse token fail, unknown '{node.text}'")
    return tokens
