from itertools import groupby

from utils.stream import StreamTokenIterator, Token, EmptyToken, reduce_token_list, SeqIterator

BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'

ReservedWords = [
    Token.Identifier, Token.Keyword, Token.Number,
    Token.String, Token.Spaces,
    Token.Symbol, Token.Operator,
]


def read_number(stream_iterator: StreamTokenIterator) -> Token:
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


def read_float_number(stream_iterator: StreamTokenIterator) -> Token:
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


def is_spaces(ch: str) -> bool:
    return index_of([' ', '\r', '\n', '\t' ], ch) != -1

def read_spaces(stream_iterator: StreamTokenIterator) -> Token:
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

def read_single_quote_string(stream_iterator: StreamTokenIterator):
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


def read_double_quote_string(stream_iterator: StreamTokenIterator):
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


def peek_str_by_list_contain(stream_iterator: StreamTokenIterator, peek_length_list: list[int], str_list: list[str], case_insensitive: bool=False):
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

def try_read_any(stream_iterator: StreamTokenIterator, fn_list: list):
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
    values = [char2index_dict[BOS_TOKEN]]
    for token in token_list:
        values += token_to_index(char2index_dict, token)
    values.append(char2index_dict[EOS_TOKEN])
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
    iterator = SeqIterator(value_list)
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
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN
] + letters + ReservedWords
fixed_length = len(fixed_marks)

TSQL_Keywords = sort_desc([
    "ADD",
    "EXTERNAL",
    "PROCEDURE",
    "ALL",
    "FETCH",
    "PUBLIC",
    "ALTER",
    "FILE",
    "RAISERROR",
    "AND",
    "FILLFACTOR",
    "READ",
    "ANY",
    "FOR",
    "READTEXT",
    "AS",
    "FOREIGN",
    "RECONFIGURE",
    "ASC",
    "FREETEXT",
    "REFERENCES",
    "AUTHORIZATION",
    "FREETEXTTABLE",
    "REPLICATION",
    "BACKUP",
    "FROM",
    "RESTORE",
    "BEGIN",
    "FULL",
    "RESTRICT",
    "BETWEEN",
    "FUNCTION",
    "RETURN",
    "BREAK",
    "GOTO",
    "REVERT",
    "BROWSE",
    "GRANT",
    "REVOKE",
    "BULK",
    "GROUP",
    "RIGHT",
    "BY",
    "HAVING",
    "ROLLBACK",
    "CASCADE",
    "HOLDLOCK",
    "ROWCOUNT",
    "CASE",
    "IDENTITY",
    "ROWGUIDCOL",
    "CHECK",
    "IDENTITY_INSERT",
    "RULE",
    "CHECKPOINT",
    "IDENTITYCOL",
    "SAVE",
    "CLOSE",
    "IF",
    "SCHEMA",
    "CLUSTERED",
    "IN",
    "SECURITYAUDIT",
    "COALESCE",
    "INDEX",
    "SELECT",
    "COLLATE",
    "INNER",
    "SEMANTICKEYPHRASETABLE",
    "COLUMN",
    "INSERT",
    "SEMANTICSIMILARITYDETAILSTABLE",
    "COMMIT",
    "INTERSECT",
    "SEMANTICSIMILARITYTABLE",
    "COMPUTE",
    "INTO",
    "SESSION_USER",
    "CONSTRAINT",
    "IS",
    "SET",
    "CONTAINS",
    "JOIN",
    "SETUSER",
    "CONTAINSTABLE",
    "KEY",
    "SHUTDOWN",
    "CONTINUE",
    "KILL",
    "SOME",
    "CONVERT",
    "LEFT",
    "STATISTICS",
    "CREATE",
    "LIKE",
    "SYSTEM_USER",
    "CROSS",
    "LINENO",
    "TABLE",
    "CURRENT",
    "LOAD",
    "TABLESAMPLE",
    "CURRENT_DATE",
    "MERGE",
    "TEXTSIZE",
    "CURRENT_TIME",
    "NATIONAL",
    "THEN",
    "CURRENT_TIMESTAMP",
    "NOCHECK",
    "TO",
    "CURRENT_USER",
    "NONCLUSTERED",
    "TOP",
    "CURSOR",
    "NOT",
    "TRAN",
    "DATABASE",
    "NULL",
    "TRANSACTION",
    "DBCC",
    "NULLIF",
    "TRIGGER",
    "DEALLOCATE",
    "OF",
    "TRUNCATE",
    "DECLARE",
    "OFF",
    "TRY_CONVERT",
    "DEFAULT",
    "OFFSETS",
    "TSEQUAL",
    "DELETE",
    "ON",
    "UNION",
    "DENY",
    "OPEN",
    "UNIQUE",
    "DESC",
    "OPENDATASOURCE",
    "UNPIVOT",
    "DISK",
    "OPENQUERY",
    "UPDATE",
    "DISTINCT",
    "OPENROWSET",
    "UPDATETEXT",
    "DISTRIBUTED",
    "OPENXML",
    "USE",
    "DOUBLE",
    "OPTION",
    "USER",
    "DROP",
    "OR",
    "VALUES",
    "DUMP",
    "ORDER",
    "VARYING",
    "ELSE",
    "OUTER",
    "VIEW",
    "END",
    "OVER",
    "WAITFOR",
    "ERRLVL",
    "PERCENT",
    "WHEN",
    "ESCAPE",
    "PIVOT",
    "WHERE",
    "EXCEPT",
    "PLAN",
    "WHILE",
    "EXEC",
    "PRECISION",
    "WITH",
    "EXECUTE",
    "PRIMARY",
    "WITHIN GROUP",
    "EXISTS",
    "PRINT",
    "WRITETEXT",
    "EXIT",
    "PROC",
])

LINQ_Keywords = sort_desc([
    "select",
    "from",
    "equals",
    "group",
    "by",
    "into",
    "orderby",
    "join",
    "in",
    "new",
    "DefaultIfEmpty",
    "Distinct",
    "ToList",
    "ToArray"
])

VOCAB_MARKS = fixed_marks + sort_desc(TSQL_Keywords + LINQ_Keywords)
VOCAB_SIZE = len(VOCAB_MARKS)
VOCAB_MARKS_CHAR2INDEX = convert_str_list_to_char2index_map(VOCAB_MARKS)
BOS_TOKEN_VALUE = VOCAB_MARKS_CHAR2INDEX[BOS_TOKEN]
EOS_TOKEN_VALUE = VOCAB_MARKS_CHAR2INDEX[EOS_TOKEN]
PAD_TOKEN_VALUE = VOCAB_MARKS_CHAR2INDEX[PAD_TOKEN]

if __name__ == "__main__":
    iterator = StreamTokenIterator("'abc ''123'''")
    token = read_single_quote_string(iterator)
    print(f"{token=}")


