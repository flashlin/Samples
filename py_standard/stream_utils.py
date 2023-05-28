import csv
import itertools
import re
from typing import Generic, TypeVar, Callable, IO
from functools import reduce
import pandas as pd
from torch.utils import data as Data
from data_utils import T
from data_utils import sort_by_len_desc, create_char2index_map, group_to_lengths


def info(text):
    print("\033[32m" + text + "\033[0m")


UNDEFINED = 'undefined'
STRING = 'string'
NUMBER = 'number'
IDENTIFIER = 'identifier'
SPACES = 'spaces'
OPERATOR = 'operator'
SYMBOL = 'symbol'
KEYWORD = 'keyword'
EMPTY = '(empty)'


class Token:
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


EmptyToken = Token(EMPTY, None, -1, -1, -1)
# T = TypeVar("T")


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


class StreamTokenIterator(Generic[T]):
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
            node = Token(UNDEFINED, " ", self.idx + count, self.line, self.col + count)
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

    def next(self, n=1) -> Token:
        count = 0
        buf = []
        while count < n:
            node = self.next_token()
            buf.append(node)
            count += 1
        if count == 0:
            return EmptyToken
        return reduce_token_list(buf[0].type, buf)

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
        token = Token(UNDEFINED, character, self.idx, self.line, self.col)
        self.buffer.append(token)
        self.buffer_len += 1
        increase_idx(character)
        return token

    def is_done(self):
        if self.idx == 0 and self.length == 0:
            return True
        return self.idx >= self.length


def read_identifier_token(stream_iterator: StreamTokenIterator) -> Token:
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
    return reduce_token_list(IDENTIFIER, buff)


def read_symbol_token(stream_iterator: StreamTokenIterator, symbols: list[str]) -> Token:
    symbols_sorted = sort_by_len_desc(symbols)
    symbols_dict = create_char2index_map(symbols_sorted)
    symbols_lens = group_to_lengths(symbols_sorted)
    for symbol_len in symbols_lens:
        ch = stream_iterator.peek_str(symbol_len)
        if ch in symbols_dict:
            token = stream_iterator.next(symbol_len)
            token.type = Token.Symbol
            return token
    return EmptyToken


def read_number_token(stream_iterator: StreamTokenIterator) -> Token:
    buff = []
    while not stream_iterator.is_done():
        token = stream_iterator.peek()
        if not token.text.isdigit():
            break
        buff.append(token)
        stream_iterator.next()
    if len(buff) == 0:
        return EmptyToken
    return reduce_token_list(NUMBER, buff)


def read_float_number_token(stream_iterator: StreamTokenIterator) -> Token:
    integer = read_number_token(stream_iterator)
    text = stream_iterator.peek_str(1)
    if text != '.':
        return integer
    dot = stream_iterator.next()
    scale = read_number_token(stream_iterator)
    if scale == EmptyToken:
        stream_iterator.prev()
        return EmptyToken
    return reduce_token_list(NUMBER, [integer, dot, scale])


def index_of(arr: list[str], search: str, case_insensitive: bool = False) -> int:
    search = search.upper() if case_insensitive else search
    for idx, item in enumerate(arr):
        item = item.upper() if case_insensitive else item
        if item == search:
            return idx
    return -1


def is_spaces(ch: str) -> bool:
    return index_of([' ', '\r', '\n', '\t'], ch) != -1


def read_spaces_token(stream_iterator: StreamTokenIterator) -> Token:
    buff = []
    while not stream_iterator.is_done():
        token = stream_iterator.peek()
        if not is_spaces(token.text):
            break
        stream_iterator.next()
        buff.append(token)
    if len(buff) == 0:
        return EmptyToken
    return reduce_token_list(SPACES, buff)


def read_single_quote_string_token(stream_iterator: StreamTokenIterator) -> Token:
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
    return reduce_token_list(STRING, buff)


def read_double_quote_string_token(stream_iterator: StreamTokenIterator) -> Token:
    if stream_iterator.peek_str(1) != '"':
        return EmptyToken
    buff = [stream_iterator.next()]
    while not stream_iterator.is_done():
        token = stream_iterator.next()
        buff.append(token)
        if token.text == '"':
            if buff[len(buff) - 1].text == '\\':
                continue
            break
    return reduce_token_list(STRING, buff)


Until_Check_Fn = Callable[[StreamTokenIterator], bool]


def read_token_until_by(stream_iterator: StreamTokenIterator, check_fn: Until_Check_Fn) -> Token:
    buff = [stream_iterator.next()]
    while not stream_iterator.is_done():
        if check_fn(stream_iterator):
            break
        token = stream_iterator.next()
        buff.append(token)
    return reduce_token_list(STRING, buff)


def read_token_until(stream_iterator: StreamTokenIterator, last_char) -> Token:
    return read_token_until_by(stream_iterator, lambda a_iter: a_iter.peek_str(1) == last_char)


class SeqIterator(Generic[T]):
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


def int_list_to_str(alist):
    return ','.join([str(n) for n in alist])


def replace_many_spaces(text):
    new_text = re.sub(' +', ' ', text)
    return new_text


def read_lines_from_file_ptr(file_ptr: IO, n_lines: int):
    while True:
        lines = list(itertools.islice(file_ptr, n_lines))
        if not lines:
            break
        lines = [line.rstrip('\n') for line in lines]
        yield lines


def read_lines_from_file(file_path: str, n_lines: int = 2):
    with open(file_path, 'r', encoding='utf-8') as sr:
        line_pairs = read_lines_from_file_ptr(sr, n_lines)
        for lines in line_pairs:
            yield lines


class CsvDataSet(Data.Dataset):
    def __init__(self, csv_file_path, chunk_size=1):
        super(CsvDataSet, self).__init__()
        self.csv_file_path = csv_file_path
        self.chunk_size = chunk_size
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            self.headers = file.readline().strip().split(',')

    def __len__(self):
        with open(self.csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            return sum(1 for _ in csv_reader) - 1  # 減去 header 的行數

    def __getitem__(self, idx):
        data_frame = pd.read_csv(self.csv_file_path, skiprows=idx, nrows=self.chunk_size)
        dict_value = data_frame.iloc[0]
        new_dict = {}
        for index, header in enumerate(self.headers):
            new_dict[header] = dict_value[index]
        return new_dict
