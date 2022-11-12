from typing import Final

from utils.tokenizr import Token, StreamIterator, read_identifier, read_float_number, \
    read_single_quote_string, try_read_any, EmptyToken, sort_desc, group_to_lengths, \
    read_keyword_fn, read_spaces, convert_str_list_to_char2index_map, convert_str_list_to_index2char_map, \
    fixed_marks, tokens_to_index_list, index_list_to_string, reduce_token_list

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
TSQL_Keywords_Lengths = group_to_lengths(TSQL_Keywords)
TSQL_Symbols: Final[list[str]] = [',', '.', '(', ')', '@', '#']
TSQL_Operators: Final[list[str]] = sort_desc(['<>', '>=', '<=', '!=', '=', '+', '-', '*', '/', '%'])
# TSQL_Operators_Lengths = [(k, list(g)) for k, g in groupby(TSQL_Operators, key=lambda x: len(x))]
TSQL_Operators_Lengths = group_to_lengths(TSQL_Operators)

def read_tsql_keyword_fn():
    return read_keyword_fn(Token.Keyword, TSQL_Keywords_Lengths, TSQL_Keywords, case_insensitive=True)

def read_symbol_fn():
    return read_keyword_fn(Token.Symbol, [1], TSQL_Symbols)

def read_operator_fn():
    return read_keyword_fn(Token.Operator, TSQL_Operators_Lengths, TSQL_Operators)

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
    return reduce_token_list(Token.Identifier, buff)

def tsql_tokenize(stream) -> list[Token]:
    tokens = []
    stream_iterator = StreamIterator(stream)

    read_fn_list = [
        read_tsql_keyword_fn(),
        read_identifier,
        read_float_number,
        read_single_quote_string,
        read_operator_fn(),
        read_symbol_fn(),
        read_spaces,
        read_tsql_identifier
    ]

    while not stream_iterator.is_done():
        token = try_read_any(stream_iterator, read_fn_list)
        if token != EmptyToken:
            tokens.append(token)
            continue
        raise Exception(f"try to tokenize fail at {stream_iterator.idx=} '{stream_iterator.peek_str(10)}'")
    return tokens


tsql_marks = fixed_marks + TSQL_Keywords
tsql_char2index_dict = convert_str_list_to_char2index_map(tsql_marks)
tsql_index2char_dict = convert_str_list_to_index2char_map(tsql_marks)
TSQL_VOCAB_SIZE = len(tsql_marks)

def tsql_encode(stream):
    tokens = tsql_tokenize(stream)
    values = tokens_to_index_list(tsql_char2index_dict, tokens)
    return values

def tsql_decode(values):
    return index_list_to_string(tsql_index2char_dict, values)
