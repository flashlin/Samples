from collections.abc import MutableMapping, Mapping

from common.io import info
from utils.linq_tokenizr import linq_tokenize
from utils.stream import Token, index_of, SeqIterator
from utils.tokenizr import LINQ_Keywords, TSQL_Keywords
from utils.data_utils import sort_desc, create_char2index_map, create_index2char_map
from utils.tsql_tokenizr import tsql_tokenize, TSQL_CHAR2INDEX_DICT, TSQL_INDEX2CHAR_DICT

BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'

Token.CreateTarget = '<create>'

LETTERS = [ch for ch in "`-=~!@#$%^&*()_+{}|[]\\:\";'<>?,./ "]
TOKEN_SRC_TYPES = [
    Token.Identifier, Token.Keyword, Token.Number,
    Token.String, Token.Spaces, Token.Symbol, Token.Operator,
]
TOKEN_TYPES = TOKEN_SRC_TYPES + [
    Token.CreateTarget
]
VOCAB_MARKS = [
                  "",
                  BOS_TOKEN,
                  EOS_TOKEN,
                  PAD_TOKEN
              ] + LETTERS + TOKEN_TYPES
VOCAB_SIZE = len(VOCAB_MARKS)
VOCAB_MARKS_CHAR2INDEX = create_char2index_map(VOCAB_MARKS)

BOS_TOKEN_VALUE = VOCAB_MARKS_CHAR2INDEX[BOS_TOKEN]
EOS_TOKEN_VALUE = VOCAB_MARKS_CHAR2INDEX[EOS_TOKEN]
PAD_TOKEN_VALUE = VOCAB_MARKS_CHAR2INDEX[PAD_TOKEN]

SCAN_TSQL_MARKS = VOCAB_MARKS + sort_desc([
    'NOLOCK', 'SELECT', 'FROM', 'JOIN', 'ON',
    'LEFT', 'RIGHT', 'OUTER', 'INNER',
    'GROUP', 'AS', 'WITH'
])
SCAN_TSQL_CHAR2INDEX_DICT = create_char2index_map(SCAN_TSQL_MARKS)


class CaseInsensitiveDict(MutableMapping):
    """
    For example,
        cid = CaseInsensitiveDict()
        cid['Accept'] = 'application/json'
        cid['aCCEPT'] == 'application/json'  # True
        list(cid) == ['Accept']  # True
    """

    def __init__(self, data=None, **kwargs):
        self._store = dict()
        if data is None:
            data = {}
        self.update(data, **kwargs)

    def __setitem__(self, key, value):
        self._store[key.lower()] = (key, value)

    def __getitem__(self, key):
        return self._store[key.lower()][1]

    def __delitem__(self, key):
        del self._store[key.lower()]

    def __iter__(self):
        return (casedkey for casedkey, mappedvalue in self._store.values())

    def __len__(self):
        return len(self._store)

    def lower_items(self):
        """Like iteritems(), but with all lowercase keys."""
        return (
            (lowerkey, keyval[1])
            for (lowerkey, keyval)
            in self._store.items()
        )

    def __eq__(self, other):
        if isinstance(other, Mapping):
            other = CaseInsensitiveDict(other)
        else:
            return NotImplemented
        # Compare insensitively
        return dict(self.lower_items()) == dict(other.lower_items())

    # Copy is required
    def copy(self):
        return CaseInsensitiveDict(self._store.values())

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, dict(self.items()))


class EmbeddingToken:
    def __init__(self, token_type: str, offset: int):
        self.token_type = token_type
        self.offset = offset

    def to_values(self):
        return [index_of(TOKEN_TYPES, self.token_type), self.offset]


def find_token_in_src(tokens: list[Token], search_token: Token) -> Token:
    search_text = search_token.text
    if search_token.type == Token.Identifier:
        search_text = search_token.text[1:-1]
    if search_token.type == Token.String:
        search_text = '"' + search_token.text[1:-1] + '"'
    for idx, token in enumerate(tokens):
        if token.text == search_text:
            return idx, token
    return None, None


class Linq2TSqlEmbedding:
    def __init__(self):
        self.token_src_types = [Token.Identifier]
        self.token_type_dict = create_index2char_map(TOKEN_TYPES)
        self.tgt_char2index_dict = CaseInsensitiveDict(create_char2index_map(SCAN_TSQL_MARKS))
        self.tgt_index2char_dict = create_index2char_map(SCAN_TSQL_MARKS)

    def encode_source(self, text):
        tokens = linq_tokenize(text)
        values = []
        for idx, token in enumerate(tokens):
            value = EmbeddingToken(token.type, idx).to_values()
            values.extend(value)
        return tokens, values

    def encode_target(self, text, src_tokens: list[Token]):
        tgt_tokens = tsql_tokenize(text)
        values = []
        for tgt_token in tgt_tokens:
            (idx, src_token) = find_token_in_src(src_tokens, tgt_token)
            if src_token is not None and self.is_token_src_type(src_token.type):
                value = EmbeddingToken(src_token.type, idx)
                values.append(value)
                continue
            value = EmbeddingToken(Token.CreateTarget, self.tgt_char2index_dict[tgt_token.text])
            values.append(value)

        tgt_values = []
        for x in values:
            tgt_values.extend(x.to_values())
        return tgt_tokens, tgt_values

    def decode(self, tgt_values, src_tokens):
        seq_iterator = SeqIterator(tgt_values)
        text = ""
        while not seq_iterator.is_done():
            token_type_value = seq_iterator.next()
            token_type = self.token_type_dict[token_type_value]
            token_value = seq_iterator.next()
            if token_type != Token.CreateTarget:
                text += src_tokens[token_value].text
                continue

            src_text = self.tgt_index2char_dict[token_value]
            tgt_value = self.tgt_char2index_dict[src_text]
            tgt_text = self.tgt_index2char_dict[tgt_value]
            text += tgt_text

        return text

    def is_token_src_type(self, search_token_type):
        for token_type in self.token_src_types:
            if token_type == search_token_type:
                return True
        return False


if __name__ == "__main__":
    linq_code = 'from tb1 in customer select tb1'
    tsql_code = 'SELECT [tb1].* FROM [customer] AS [tb1] WITH(NOLOCK)'
    emb = Linq2TSqlEmbedding()
    the_src_tokens, the_src_values = emb.encode_source(linq_code)
    print(f"{the_src_values=}")
    the_tgt_tokens, the_tgt_values = emb.encode_target(tsql_code, the_src_tokens)
    info(f"{the_tgt_values=}")
    s2 = emb.decode(the_tgt_values, the_src_tokens)
    print(s2)
