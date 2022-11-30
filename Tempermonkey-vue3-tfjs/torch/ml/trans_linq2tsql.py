import re

from common.io import info_error
from my_model import encode_tokens, decode_to_text
from utils.data_utils import sort_desc, create_char2index_map, create_index2char_map
from utils.stream import StreamTokenIterator, Token, EmptyToken, read_identifier_token, reduce_token_list, \
    read_double_quote_string_token, read_spaces_token, read_symbol_token, read_token_until


class LinqToSqlVocab:
    def __init__(self):
        self.symbols = symbols = '. [ ] { } += + - * / , =='.split(' ')
        common_symbols = '1 2 3 4 5 6 7 8 9 0 <unk> <bos> <eos> <pad>'.split(' ') + [' ']
        linq_spec = 'from in select new join on equals contains'.split(' ')
        linq_symbols = sort_desc(common_symbols + symbols + linq_spec)
        tsql_spec = '@tb_as @tb @fd_as @fd @str @number'.split(' ')
        tsql_symbols = sort_desc(common_symbols + symbols + tsql_spec)
        self.shared_symbols = shared_symbols = sort_desc(common_symbols + symbols + linq_symbols + tsql_symbols)
        self.char2index = create_char2index_map(shared_symbols)
        self.index2char = create_index2char_map(shared_symbols)

    def get_size(self):
        return len(self.shared_symbols)

    def encode_tokens(self, tokens: [str]) -> [int]:
        char2index = self.char2index
        var_re = re.compile(r'(@\w.+)(\d+)')
        buff = [char2index['<bos>']]
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
        buff.append(char2index['<eos>'])
        return buff

    def encode_number_token(self, token):
        return [self.char2index[n] for n in token]

    def decode_values(self, values: [int]) -> [str]:
        buff = []
        for value in values:
            buff.append(self.index2char[value])
        return buff

    @staticmethod
    def read_variable_token(stream_iter: StreamTokenIterator) -> Token:
        if stream_iter.peek_str(1) != '@':
            return EmptyToken
        at_token = stream_iter.next()
        token = read_identifier_token(stream_iter)
        return reduce_token_list('variable', [at_token, token])

    def encode_to_tokens(self, line) -> [str]:
        stream_iter = StreamTokenIterator(line)
        buff = []
        while not stream_iter.is_done():
            token = LinqToSqlVocab.read_variable_token(stream_iter)
            if token != EmptyToken:
                buff.append(token.text)
                continue
            token = read_double_quote_string_token(stream_iter)
            if token != EmptyToken:
                buff.append(token.text)
                continue
            token = read_spaces_token(stream_iter)
            if token != EmptyToken:
                buff.append(' ')
                continue
            token = read_symbol_token(stream_iter, self.symbols)
            if token != EmptyToken:
                buff.append(token.text)
                continue
            token = read_identifier_token(stream_iter)
            if token != EmptyToken:
                buff.append(token.text)
                continue

            text = read_token_until(stream_iter, ' ').text
            buff.append(text)
        return buff

    def encode(self, text: str) -> [int]:
        tokens = self.encode_to_tokens(text)
        return self.encode_tokens(tokens)

    def get_value(self, char: str) -> int:
        return self.char2index[char]
