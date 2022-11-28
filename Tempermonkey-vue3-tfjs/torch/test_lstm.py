import torch
from torch import nn

from common.io import info
from my_model import line_to_tokens
from utils.stream import StreamTokenIterator, read_double_quote_string, read_until, read_identifier, EmptyToken, \
    read_symbol
from utils.data_utils import sort_desc, group_to_lengths, create_char2index_map

"""
src = 'from tb1     in customer select tb1     . name'
tgt = 'from @tb_as1 in @tb1     select @tb_as1 . @fd1'

src = 'from tb1     in'
pre = '<bos>'
tgt = 'from @tb_as1 in'

src = 'tb1     in customer'
pre = 'from'
tgt = '@tb_as1 in @tb1'

src = 'in customer select'
pre = 'from in'
tgt = 'in @tb1     select'

src = 'customer select tb1'
pre = 'from in customer'
tgt = '@tb1     select @tb_as1'

src = 'select tb1     .'
pre = 'in customer select'
tgt = 'select @tb_as1 .'

src = 'tb1     . name'
pre = 'customer select tb1'
tgt = '@tb_as1 . @fd1'

src = '. name <eos>'
pre = 'select tb1 .'
tgt = '. @fd1 <eos>'
"""

src = 'from tb1 in customer select tb1.name'
pre = ''
tgt = 'from @tb_as1 in @tb1 select @tb_as1 . @fd1'
# lstm = nn.LSTM()

symbols = '. [ ] { } += + - * / ,'
symbols = symbols.split(' ')


def linq_to_tokens(line):
    stream_iter = StreamTokenIterator(line)
    buff = []
    prev_ch = None
    while not stream_iter.is_done():
        ch = stream_iter.peek_str(1)
        if ch == '"':
            prev_ch = read_double_quote_string(stream_iter).text
            buff.append(prev_ch)
            continue
        if ch == ' ':
            if prev_ch == ' ':
                prev_ch = stream_iter.next().text
                continue
            prev_ch = stream_iter.next().text
            buff.append(prev_ch)
            continue
        token = read_symbol(stream_iter, symbols)
        if token != EmptyToken:
            prev_ch = token.text
            buff.append(token.text)
            continue
        token = read_identifier(stream_iter)
        if token != EmptyToken:
            prev_ch = token.text
            buff.append(token.text)
            continue

        text = read_until(stream_iter, ' ').text
        prev_ch = text
        buff.append(text)
    return buff


src_tokens = linq_to_tokens(src)
print(f"{src_tokens=}")

# max_size = 297
# embedding = nn.Embedding(max_size + 1, 3, padding_idx=0)
# inputs = torch.tensor([  1, 102, 243, 105,  97, 101,  21,  22,  23,   6,   7,  16,  22,   0,
#         105,  97, 102, 297, 105,  97, 101,  21,  22,  23,   6,   7,  16,  22,
#          21,   0, 105,  97, 102, 221, 105,  97, 101,  21,  22,  23,   6,   7,
#          16,  22,   0, 105,  97, 102, 296, 105,  97, 101,  21,  22,  23,   6,
#           7,  16,  22,   0, 106,  95, 101,  40,   3,  21,  22,  42,   3,  15,
#           7,   0, 105,  97, 102, 244, 105,  97, 102, 282, 101,  35,  20,  17,
#          23,  18,   0, 105,  97, 102, 168, 105,  97, 102, 282, 101,  35,  20,
#          17,  23,  18,   0, 106,  95, 101,  39,   7,  27,   0, 105,  97, 102,
#         192, 105,  97, 102, 282, 106,  81, 101,  13,   7,  27,   0, 107,  67,
#         102, 282, 101,  35,  20,  17,  23,  18,   0, 106,  95, 101,  39,   7,
#          27,   0, 106,  94, 101,  24,   3,  14,  23,   7,  21,   0, 107,  67,
#         102, 282, 101,  35,  20,  17,  23,  18,   0, 106,  95, 102, 169, 106,
#          77, 106,  78, 106,  82, 105,  98,   2,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0])
# info(f" {inputs.shape=}")
# info(f" {max(inputs)=}")
# assert (inputs <= max_size).all(), "target: {} invalid".format(inputs)
# t = embedding(inputs) #.view(1, 1, -1)
# info(f" {t=}")
