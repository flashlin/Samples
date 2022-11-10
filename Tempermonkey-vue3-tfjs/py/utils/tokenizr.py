from functools import reduce
from typing import TypeVar, Generic
from itertools import groupby

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

ReservedWords = [
    Token.Identifier, Token.Keyword, Token.Number,
    Token.String, Token.Spaces,
    Token.Symbol, Token.Operator,
]
EmptyToken = Token(Token.Empty, None, -1, -1, -1)
T = TypeVar("T")

class Iterator(Generic[T]):
    def __init__(self, stream: list[T]):
        self.stream = stream
        self.length = len(stream)
        self.idx = 0
        self.buffer: list[T] = []
        self.buffer_len = 0

    def peek(self, n=1):
        node = self.next(n)
        self.prev(n)
        return node

    def prev(self, n=1):
        node = None
        count = 0
        while count < n:
            node = self.prev_token()
            count += 1
        return node

    def prev_token(self):
        if self.idx - 1 < 0:
            return None
        self.idx -= 1
        return self.buffer[self.idx]

    def next(self, n=1):
        node = None
        count = 0
        while count < n:
            node = self.next_token()
            count += 1
        return node

    def next_token(self):
        if self.idx >= self.length:
            return EmptyToken
        if self.idx < self.buffer_len:
            buffer_node = self.buffer[self.idx]
            self.idx += 1
            return buffer_node
        character = self.stream[self.idx]
        self.buffer.append(character)
        self.buffer_len += 1
        self.idx += 1
        return character

    def is_done(self):
        if self.idx == 0 and self.length == 0:
            return True
        return self.idx >= self.length

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

def is_spaces(ch: str) -> bool:
    return index_of([' ', '\r', '\n', '\t' ], ch) != -1

def read_spaces(stream_iterator: StreamIterator) -> Token:
    buff = []
    while not stream_iterator.is_done():
        token = stream_iterator.peek()
        if not is_spaces(token.text):
            break
        stream_iterator.next()
        buff.append(token)
    if len(buff) == 0:
        return EmptyToken
    return reduce_token_list(Token.Spaces, buff)

def read_single_quote_string(stream_iterator: StreamIterator):
    if stream_iterator.peek_str(1) != "'":
        return EmptyToken
    buff = [stream_iterator.next()]
    while not stream_iterator.is_done():
        token = stream_iterator.peek()
        if token.text == "'":
            stream_iterator.next()
            buff.append(token)
            token2 = stream_iterator.peek()
            if token2.text == "'":
                buff.append(token2)
                stream_iterator.next()
                continue
            break
        stream_iterator.next()
        buff.append(token)
    return reduce_token_list(Token.String, buff)


def read_double_quote_string(stream_iterator: StreamIterator):
    if stream_iterator.peek_str(1) != '"':
        return EmptyToken
    buff = [stream_iterator.next()]
    while not stream_iterator.is_done():
        token = stream_iterator.next()
        buff.append(token)
        if token.text == '"':
            if buff[len(buff)-1].text == '\\':
                continue
            break
    return reduce_token_list(Token.String, buff)


def sort_desc(arr: list[str]) -> list[str]:
    arr.sort(key=lambda x: len(x))
    return arr[::-1]


def group_to_lengths(arr_sorted: list[str]):
    return [k for k, g in groupby(arr_sorted, key=lambda x: len(x))]


def index_of(arr: list[str], search: str, case_insensitive: bool=False) -> int:
    search = search.upper() if case_insensitive else search
    for idx, item in enumerate(arr):
        item = item.upper() if case_insensitive else item
        if item == search:
            return idx
    return -1


def peek_str_by_list_contain(stream_iterator: StreamIterator, peek_length_list: list[int], str_list: list[str], case_insensitive: bool=False):
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


def read_keyword_fn(token_type: str, length_list, keyword_list, case_insensitive: bool=False):
    def fn(stream_iterator):
        hint_length, keyword_value = peek_str_by_list_contain(stream_iterator, length_list, keyword_list, case_insensitive)
        buff = read_token_list_by_length(stream_iterator, hint_length)
        if not hint_length > 0:
            return EmptyToken
        token = reduce_token_list(token_type, buff)
        token.value = keyword_value
        return token
    return fn

def try_read_any(stream_iterator: StreamIterator, fn_list: list):
    for fn in fn_list:
        token = fn(stream_iterator)
        if token != EmptyToken:
            return token
    return EmptyToken

def is_string_token_type(token_type: str):
    return index_of([Token.Identifier, Token.Number, Token.String], token_type) != -1

def token_to_index(char2index_dict, token: Token):
    values = [char2index_dict[token.type]]
    if is_string_token_type(token.type):
        for ch in [ch for ch in token.value]:
            values.append(char2index_dict[ch])
        values.append(char2index_dict[''])
        return values
    values.append(char2index_dict[token.value])
    return values


def tokens_to_index_list(char2index_dict, token_list: list[Token]):
    values = [char2index_dict['<begin>']]
    for token in token_list:
        values += token_to_index(char2index_dict, token)
    values.append(char2index_dict['<end>'])
    return values

def read_until_zero(index2char_dict, iterator):
    text = ""
    while not iterator.is_done():
        value = iterator.next()
        if value == 0:
            break
        text += index2char_dict[value]
    return text

def index_list_to_string(index2char_dict, value_list: list[int]):
    text = ""
    iterator = Iterator(value_list)
    while not iterator.is_done():
        value = iterator.next()
        token_type = index2char_dict[value]
        if token_type.startswith('<') and token_type.endswith('>'):
            continue
        if is_string_token_type(token_type):
            text += read_until_zero(index2char_dict, iterator)
            continue
        text += index2char_dict[iterator.next()]
    return text

def convert_str_list_to_char2index_map(str_list: list[str]):
    dictionary = {}
    for idx, key in enumerate(str_list):
        dictionary[key] = idx
    return dictionary

def convert_str_list_to_index2char_map(str_list: list[str]):
    dictionary = {}
    for idx, key in enumerate(str_list):
        dictionary[idx] = key
    return dictionary


letters = [ch for ch in
           "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`1234567890-=~!@#$%^&*()_+{}|[]\\:\";'<>?,./ \n\r\t"]
fixed_marks = [
    "",
    "<begin>",
    "<end>",
] + letters + ReservedWords
fixed_length = len(fixed_marks)


if __name__ == "__main__":
    iterator = StreamIterator("'abc ''123'''")
    token = read_single_quote_string(iterator)
    print(f"{token=}")