from enum import Enum

from data_utils import sort_by_len_desc, group_to_lengths, write_dict_to_file, load_dict_from_file
from stream_utils import Token, StreamTokenIterator, read_identifier_token, read_float_number_token, \
    read_single_quote_string_token, read_spaces_token, KEYWORD, OPERATOR, SYMBOL, EmptyToken, reduce_token_list, \
    IDENTIFIER, NoneToken
from tokenizr_utils import read_keyword_fn, try_read_any
from typing import Final

from vocabulary_utils import WordVocabulary

TSQL_KEYWORDS = sort_by_len_desc([
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

TSQL_KEYWORDS_LENGTHS = group_to_lengths(TSQL_KEYWORDS)


def read_tsql_keyword_fn():
    return read_keyword_fn(KEYWORD, TSQL_KEYWORDS_LENGTHS, TSQL_KEYWORDS, case_insensitive=True)


TSQL_OPERATORS: Final[list[str]] = sort_by_len_desc(['<>', '>=', '<=', '!=', '=', '+', '-', '*', '/', '%'])
TSQL_OPERATORS_LENGTHS = group_to_lengths(TSQL_OPERATORS)


def read_operator_fn():
    return read_keyword_fn(OPERATOR, TSQL_OPERATORS_LENGTHS, TSQL_OPERATORS)


TSQL_SYMBOLS: Final[list[str]] = [',', '.', '(', ')', '@', '#']


def read_symbol_fn():
    return read_keyword_fn(SYMBOL, [1], TSQL_SYMBOLS)


def read_tsql_identifier(stream_iterator):
    if stream_iterator.peek_str(1) != "[":
        return EmptyToken
    buff = [stream_iterator.next()]
    while not stream_iterator.is_done():
        token = stream_iterator.peek()
        if token.text == "]":
            stream_iterator.next()
            buff.append(token)
            break
        stream_iterator.next()
        buff.append(token)
    return reduce_token_list(IDENTIFIER, buff)


def skip_spaces(stream_iterator: StreamTokenIterator):
    read_spaces_token(stream_iterator)
    return NoneToken


def tsql_tokenize(stream) -> list[Token]:
    tokens = []
    stream_iterator = StreamTokenIterator(stream)

    read_fn_list = [
        read_tsql_keyword_fn(),
        read_identifier_token,
        read_float_number_token,
        read_single_quote_string_token,
        read_operator_fn(),
        read_symbol_fn(),
        skip_spaces,
        read_tsql_identifier
    ]

    while not stream_iterator.is_done():
        token = try_read_any(stream_iterator, read_fn_list)
        if token == NoneToken:
            continue
        if token != EmptyToken:
            tokens.append(token)
            continue
        raise Exception(f"try to tokenize fail at {stream_iterator.idx=} '{stream_iterator.peek_str(10)}'")
    return tokens


class SqlVocabulary:
    def __init__(self):
        self.vocab = WordVocabulary()

    @property
    def SOS_index(self):
        return self.vocab.SOS_index

    @property
    def EOS_index(self):
        return self.vocab.EOS_index

    @property
    def PAD_index(self):
        return self.vocab.PAD_index

    def __len__(self):
        return len(self.vocab)

    def encode_many_words(self, words: list[str]):
        return self.vocab.encode_many_words(words)

    def decode_value_list(self, values: list[int], is_show: bool = False):
        return self.vocab.decode_value_list(values, isShow=is_show)

    def save(self, file_path: str):
        write_dict_to_file(self.vocab.to_serializable(), file_path)

    def load(self, file_path: str):
        vocab_dict = load_dict_from_file(file_path)
        self.vocab.from_serializable(vocab_dict['token_to_idx'])

    def encode_infer_text(self, text):
        tokens = tsql_tokenize(text)
        words = [token.text for token in tokens]
        return [self.vocab.SOS_index] + self.remove_enum(self.vocab.encode_many_words(words)) + [self.vocab.EOS_index]

    @staticmethod
    def remove_enum(value_list):
        return [item.value if isinstance(item, Enum) else item for item in value_list]
