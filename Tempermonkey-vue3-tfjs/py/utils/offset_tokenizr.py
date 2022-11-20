from utils.stream import Token
from utils.tokenizr import sort_desc, LINQ_Keywords, create_char2index_map

BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'

ReservedWords = [
    Token.Identifier, Token.Keyword, Token.Number,
    Token.String, Token.Spaces,
    Token.Symbol, Token.Operator,
]

letters = [ch for ch in
           "`-=~!@#$%^&*()_+{}|[]\\:\";'<>?,./ "]
fixed_marks = [
                  "",
                  BOS_TOKEN,
                  EOS_TOKEN,
                  PAD_TOKEN
              ] + letters + ReservedWords
fixed_length = len(fixed_marks)

VOCAB_MARKS = fixed_marks
VOCAB_SIZE = len(VOCAB_MARKS)
VOCAB_MARKS_CHAR2INDEX = create_char2index_map(VOCAB_MARKS)

BOS_TOKEN_VALUE = VOCAB_MARKS_CHAR2INDEX[BOS_TOKEN]
EOS_TOKEN_VALUE = VOCAB_MARKS_CHAR2INDEX[EOS_TOKEN]
PAD_TOKEN_VALUE = VOCAB_MARKS_CHAR2INDEX[PAD_TOKEN]

SCAN_LINQ_MARKS = letters + sort_desc(LINQ_Keywords)

if __name__ == "__main__":
    pass
