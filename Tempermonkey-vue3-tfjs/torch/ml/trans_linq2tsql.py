import re

from common.io import info_error
from utils.data_utils import sort_desc, create_char2index_map, create_index2char_map
from utils.stream import StreamTokenIterator, Token, EmptyToken, read_identifier_token, reduce_token_list, \
    read_double_quote_string_token, read_spaces_token, read_symbol_token, read_token_until, read_float_number_token, \
    read_number_token


class LinqToSqlVocab:
    def __init__(self):
        self.symbols = symbols = '. ( ) [ ] { } += + - * / , =='.split(' ')
        common_symbols = '1 2 3 4 5 6 7 8 9 0 <unk> <bos> <eos> <pad>'.split(' ') + [' ']
        linq_keywords = 'from in select new join on equals contains'.split(' ')
        tsql_keywords = 'SELECT FROM WITH NOLOCK AS JOIN LEFT RIGHT ON GROUP BY TOP DESC'.split(' ')
        linq_symbols = sort_desc(common_symbols + symbols + linq_keywords)
        tsql_spec = '@tb_as @tb @fd_as @fd @str @n'.split(' ')
        tsql_symbols = sort_desc(common_symbols + symbols + tsql_spec)
        self.keywords = create_char2index_map(linq_keywords + tsql_keywords)
        self.shared_symbols = shared_symbols = sort_desc(common_symbols + symbols + linq_symbols + tsql_symbols)
        self.char2index = create_char2index_map(shared_symbols)
        self.index2char = create_index2char_map(shared_symbols)
        self.padding_idx = self.get_value('<pad>')
        self.bos_idx = self.get_value('<bos>')
        self.eos_idx = self.get_value('<eos>')

    def get_size(self):
        return len(self.shared_symbols)

    def encode_tokens(self, tokens: [str], add_bos_eos=True) -> [int]:
        char2index = self.char2index
        var_re = re.compile(r'(@\w+)(\d+)')
        buff = []
        if add_bos_eos:
            buff.append(char2index['<bos>'])
        unk_tokens = {}
        for token in tokens:
            match = var_re.match(token)
            if match:
                name = match.group(1)
                num = match.group(2)
                buff.append(char2index[name])
                buff.append(char2index[num])
                continue
            if token not in char2index:
                unk_number = len(unk_tokens) + 1
                if token in unk_tokens:
                    unk = unk_tokens[token]
                else:
                    try:
                        unk_num_values = self.encode_number_token(str(unk_number))
                        unk = [char2index['<unk>']] + unk_num_values
                        unk_tokens[token] = unk
                    except Exception as e:
                        info_error(f' {tokens=} {token=} {e}')
                        raise
                buff.extend(unk)
                continue
            buff.append(char2index[token])
        if add_bos_eos:
            buff.append(char2index['<eos>'])
        return buff

    def encode_number_token(self, token):
        return [self.char2index[n] for n in token]

    def decode_values(self, values: [int]) -> [str]:
        def is_ignore(text):
            return text.startswith('<') and text.endswith('>')
        buff = []
        for value in values:
            token_text = self.index2char[value]
            if not is_ignore(token_text):
                buff.append(token_text)
        return buff

    def decode(self, values: [int]) -> str:
        return ''.join(self.decode_values(values))

    @staticmethod
    def read_variable_token(stream_iter: StreamTokenIterator) -> Token:
        if stream_iter.peek_str(1) != '@':
            return EmptyToken
        at_token = stream_iter.next()
        token = read_identifier_token(stream_iter)
        return reduce_token_list('variable', [at_token, token])

    @staticmethod
    def read_spec_identifier_token(stream_iter: StreamTokenIterator) -> Token:
        if stream_iter.peek_str(1) != '[':
            return EmptyToken
        start_token = stream_iter.next()
        ident = read_token_until(stream_iter, ']')
        end_token = stream_iter.next()
        return reduce_token_list(Token.Identifier, [start_token, ident, end_token])

    def parse_to_tokens(self, line) -> [Token]:
        stream_iter = StreamTokenIterator(line)
        buff = []
        while not stream_iter.is_done():
            token = LinqToSqlVocab.read_spec_identifier_token(stream_iter)
            if token != EmptyToken:
                buff.append(token)
                continue
            token = LinqToSqlVocab.read_variable_token(stream_iter)
            if token != EmptyToken:
                buff.append(token)
                continue
            token = read_double_quote_string_token(stream_iter)
            if token != EmptyToken:
                buff.append(token)
                continue
            token = read_spaces_token(stream_iter)
            if token != EmptyToken:
                buff.append(token)
                continue
            token = read_symbol_token(stream_iter, self.symbols)
            if token != EmptyToken:
                buff.append(token)
                continue
            token = read_identifier_token(stream_iter)
            if token != EmptyToken:
                buff.append(token)
                continue
            token = read_float_number_token(stream_iter)
            if token != EmptyToken:
                buff.append(token)
                continue
            token = read_number_token(stream_iter)
            if token != EmptyToken:
                buff.append(token)
                continue
            token = read_token_until(stream_iter, ' ')
            buff.append(token)
        return buff

    @staticmethod
    def add_to_dict(dict, key):
        if key in dict:
            return dict[key]
        new_index = len(dict) + 1
        dict[key] = new_index
        return new_index

    def encode_to_words(self, line) -> [str]:
        identifiers = {}
        numbers = {}
        strings = {}
        buff = []
        for token in self.parse_to_tokens(line):
            if token.text in self.keywords:
                buff.append(token.text)
                continue
            if token.type == Token.Identifier:
                idx = self.add_to_dict(identifiers, token.text)
                buff.append(f'<identifier{idx}>')
                continue
            if token.type == 'variable':
                buff.append(token.text)
                continue
            if token.type == Token.String:
                idx = self.add_to_dict(strings, token.text)
                buff.append(f'<s{idx}>')
                continue
            if token.type == Token.Number:
                idx = self.add_to_dict(numbers, token.text)
                buff.append(f'<n{idx}>')
                continue
            if token.type == Token.Spaces:
                buff.append(f' ')
                continue
            buff.append(token.text)
        return buff

    def encode(self, text: str, add_bos_eos=True) -> [int]:
        tokens = self.encode_to_texts(text)
        return self.encode_tokens(tokens, add_bos_eos)

    def get_value(self, char: str) -> int:
        return self.char2index[char]


class EnglishVocab:
    def __init__(self):
        self.symbols = symbols = '. [ ] { } = _ + - * / , > < ! @ \' " & ( )'.split(' ')
        common_symbols = '1 2 3 4 5 6 7 8 9 0 <unk> <bos> <eos> <pad>'.split(' ') + [' ']
        spec = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.shared_symbols = shared_symbols = sort_desc(common_symbols + symbols + spec)
        self.char2index = create_char2index_map(shared_symbols)
        self.index2char = create_index2char_map(shared_symbols)
        self.padding_idx = self.get_value('<pad>')
        self.bos_idx = self.get_value('<bos>')
        self.eos_idx = self.get_value('<eos>')

    def get_size(self):
        return len(self.shared_symbols)

    def decode(self, values: [int]) -> str:
        buf = []
        for idx in values:
            buf.append(self.index2char[idx])
        return ''.join(buf)

    def encode(self, text: str) -> [int]:
        buf = [self.bos_idx]
        for ch in text:
            buf.append(self.char2index[ch])
        buf.append(self.eos_idx)
        return buf

    def get_value(self, char: str) -> int:
        return self.char2index[char]
