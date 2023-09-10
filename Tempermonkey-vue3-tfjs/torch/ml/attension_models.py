import numpy as np
import torch
from torch import nn

from common.io import info


# 计算角度：pos * 1/(10000^(2i/d))
def get_angles(pos, i, d_model):
    # 2*(i//2)保证了2i，这部分计算的是1/10000^(2i/d)
    angle_rates = 1 / np.power(10000, 2 * (i // 2) / np.float32(d_model))  # => [1, 512]
    return pos * angle_rates  # [50,1]*[1,512]=>[50, 512]


# np.arange()函数返回一个有终点和起点的固定步长的排列，如[1,2,3,4,5]，起点是1，终点是5，步长为1
# 注意：起点终点是左开右闭区间，即start=1,end=6，才会产生[1,2,3,4,5]
# 只有一个参数时，参数值为终点，起点取默认值0，步长取默认值1。
def positional_encoding(position, d_model):  # d_model是位置编码的长度，相当于position encoding的embedding_dim？
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],  # [50, 1]
                            np.arange(d_model)[np.newaxis, :],  # [1, d_model=512]
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # 2i
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # 2i+2

    pos_encoding = angle_rads[np.newaxis, ...]  # [50,512]=>[1,50,512]
    return torch.tensor(pos_encoding, dtype=torch.float32)


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    #計算注意力權重
    q, k, v 必须具有匹配的前置维度。 且dq=dk
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    #雖然 mask 根据其类型（填充或前瞻）有不同的形状，
    #但是 mask 必須能進行廣播轉換以便求和。

    #参数:
        q: 请求的形状 == (..., seq_len_q, depth)
        k: 主键的形状 == (..., seq_len_k, depth)
        v: 数值的形状 == (..., seq_len_v, depth_v)  seq_len_k = seq_len_v
        mask: Float 张量，其形状能转换成
              (..., seq_len_q, seq_len_k)。默认为None。
    #返回值:
        #输出，注意力权重
    """
    # matmul(a,b)矩阵乘:a b的最后2个维度要能做乘法，即a的最后一个维度值==b的倒数第2个纬度值，
    # 除此之外，其他维度值必须相等或为1（为1时会广播）
    matmul_qk = torch.matmul(q, k.transpose(-1, -2))  # 矩阵乘 =>[..., seq_len_q, seq_len_k]

    # 缩放matmul_qk
    dk = torch.tensor(k.shape[-1], dtype=torch.float32)  # k的深度dk，或叫做depth_k
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)  # [..., seq_len_q, seq_len_k]

    # 将 mask 加入到缩放的张量上(重要！)
    if mask is not None:  # mask: [b, 1, 1, seq_len]
        # mask=1的位置是pad，乘以-1e9（-1*10^9）成为负无穷，经过softmax后会趋于0
        scaled_attention_logits += (mask * -1e9)

    # softmax 在最后一个轴（seq_len_k）上归一化
    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)  # [..., seq_len_q, seq_len_k]

    output = torch.matmul(attention_weights, v)  # =>[..., seq_len_q, depth_v]
    return output, attention_weights  # [..., seq_len_q, depth_v], [..., seq_len_q, seq_len_k]


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0  # 因为输入要被（平均？）split到不同的head

        self.depth = d_model // self.num_heads  # 512/8=64，所以在scaled dot-product atten中dq=dk=64,dv也是64

        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)

        self.final_linear = torch.nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):  # x [b, seq_len, d_model]
        x = x.view(batch_size, -1, self.num_heads,
                   self.depth)  # [b, seq_len, d_model=512]=>[b, seq_len, num_head=8, depth=64]
        return x.transpose(1, 2)  # [b, seq_len, num_head=8, depth=64]=>[b, num_head=8, seq_len, depth=64]

    def forward(self, q, k, v, mask):  # q=k=v=x [b, seq_len, embedding_dim] embedding_dim其实也=d_model
        batch_size = q.shape[0]

        q = self.wq(q)  # =>[b, seq_len, d_model]
        k = self.wk(k)  # =>[b, seq_len, d_model]
        v = self.wv(v)  # =>[b, seq_len, d_model]

        q = self.split_heads(q, batch_size)  # =>[b, num_head=8, seq_len, depth=64]
        k = self.split_heads(k, batch_size)  # =>[b, num_head=8, seq_len, depth=64]
        v = self.split_heads(v, batch_size)  # =>[b, num_head=8, seq_len, depth=64]

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # => [b, num_head=8, seq_len_q, depth=64], [b, num_head=8, seq_len_q, seq_len_k]

        scaled_attention = scaled_attention.transpose(1, 2)  # =>[b, seq_len_q, num_head=8, depth=64]
        # 转置操作让张量存储结构扭曲，直接使用view方法会失败，可以使用reshape方法
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)  # =>[b, seq_len_q, d_model=512]

        output = self.final_linear(concat_attention)  # =>[b, seq_len_q, d_model=512]
        return output, attention_weights  # [b, seq_len_q, d_model=512], [b, num_head=8, seq_len_q, seq_len_k]


def point_wise_feed_forward_network(d_model, dff):
    """
    點式前饋網路
    :param d_model:
    :param dff:
    :return:
    """
    feed_forward_net = torch.nn.Sequential(
        torch.nn.Linear(d_model, dff),  # [b, seq_len, d_model]=>[b, seq_len, dff=2048]
        torch.nn.ReLU(),
        torch.nn.Linear(dff, d_model),  # [b, seq_len, dff=2048]=>[b, seq_len, d_model=512]
    )
    return feed_forward_net


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)  # 多头注意力（padding mask）(self-attention)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    # x [b, inp_seq_len, embedding_dim] embedding_dim其实也=d_model
    # mask [b,1,1,inp_seq_len]
    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # =>[b, seq_len, d_model]
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # 残差&层归一化 =>[b, seq_len, d_model]

        ffn_output = self.ffn(out1)  # =>[b, seq_len, d_model]
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # 残差&层归一化 =>[b, seq_len, d_model]

        return out2  # [b, seq_len, d_model]


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model,
                                       num_heads)  # masked的多头注意力（look ahead mask 和 padding mask）(self-attention)
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # 多头注意力（padding mask）(encoder-decoder attention)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm3 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)
        self.dropout3 = torch.nn.Dropout(rate)

    # x [b, targ_seq_len, embedding_dim] embedding_dim其实也=d_model=512
    # look_ahead_mask [b, 1, targ_seq_len, targ_seq_len] 这里传入的look_ahead_mask应该是已经结合了look_ahead_mask和padding mask的mask
    # enc_output [b, inp_seq_len, d_model]
    # padding_mask [b, 1, 1, inp_seq_len]
    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x,
                                               look_ahead_mask)  # =>[b, targ_seq_len, d_model], [b, num_heads, targ_seq_len, targ_seq_len]
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)  # 残差&层归一化 [b, targ_seq_len, d_model]

        # Q: receives the output from decoder's first attention block，即 masked multi-head attention sublayer
        # K V: V (value) and K (key) receive the encoder output as inputs
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output,
                                               padding_mask)  # =>[b, targ_seq_len, d_model], [b, num_heads, targ_seq_len, inp_seq_len]
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)  # 残差&层归一化 [b, targ_seq_len, d_model]

        ffn_output = self.ffn(out2)  # =>[b, targ_seq_len, d_model]
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)  # 残差&层归一化 =>[b, targ_seq_len, d_model]

        return out3, attn_weights_block1, attn_weights_block2
        # [b, targ_seq_len, d_model], [b, num_heads, targ_seq_len, targ_seq_len], [b, num_heads, targ_seq_len, inp_seq_len]


class Encoder(torch.nn.Module):
    def __init__(self,
                 num_layers,  # N个encoder layer
                 d_model,
                 num_heads,
                 dff,  # 点式前馈网络内层fn的维度
                 input_vocab_size,  # 输入词表大小（源语言（法语））
                 maximun_position_encoding,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = torch.nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model)
        self.pos_encoding = positional_encoding(maximun_position_encoding,
                                                d_model)  # =>[1, max_pos_encoding, d_model=512]

        # self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate).cuda() for _ in range(num_layers)] # 不行
        self.enc_layers = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])

        self.dropout = torch.nn.Dropout(rate)

    # x [b, inp_seq_len]
    # mask [b, 1, 1, inp_sel_len]
    def forward(self, x, mask):
        device = next(self.parameters()).device
        x = x.to(device)

        inp_seq_len = x.shape[-1]

        # adding embedding and position encoding
        x = self.embedding(x)  # [b, inp_seq_len]=>[b, inp_seq_len, d_model]
        # 縮放 embedding 原始論文的3.4 節有提到： In the embedding layers, we multiply those weights by \sqrt{d_model}.
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        pos_encoding = self.pos_encoding[:, :inp_seq_len, :]
        pos_encoding = pos_encoding.to(device)
        x += pos_encoding  # [b, inp_seq_len, d_model]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)  # [b, inp_seq_len, d_model]=>[b, inp_seq_len, d_model]
        return x  # [b, inp_seq_len, d_model]


class Decoder(torch.nn.Module):
    def __init__(self,
                 num_layers,  # N个encoder layer
                 d_model,
                 num_heads,
                 dff,  # 点式前馈网络内层fn的维度
                 target_vocab_size,  # target词表大小（目标语言（英语））
                 maximun_position_encoding,
                 rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = torch.nn.Embedding(num_embeddings=target_vocab_size, embedding_dim=d_model)
        self.pos_encoding = positional_encoding(maximun_position_encoding,
                                                d_model)  # =>[1, max_pos_encoding, d_model=512]

        # self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate).cuda() for _ in range(num_layers)] # 不行
        self.dec_layers = torch.nn.ModuleList([DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])

        self.dropout = torch.nn.Dropout(rate)

    # x [b, targ_seq_len]
    # look_ahead_mask [b, 1, targ_seq_len, targ_seq_len] 这里传入的look_ahead_mask应该是已经结合了look_ahead_mask和padding mask的mask
    # enc_output [b, inp_seq_len, d_model]
    # padding_mask [b, 1, 1, inp_seq_len]
    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        device = next(self.parameters()).device
        x = x.to(device)
        targ_seq_len = x.shape[-1]

        attention_weights = {}

        # adding embedding and position encoding
        x = self.embedding(x)  # [b, targ_seq_len]=>[b, targ_seq_len, d_model]
        # 缩放 embedding 原始论文的3.4节有提到： In the embedding layers, we multiply those weights by \sqrt{d_model}.
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        # x += self.pos_encoding[:, :targ_seq_len, :]  # [b, targ_seq_len, d_model]
        pos_encoding = self.pos_encoding[:, :targ_seq_len, :]  # [b, targ_seq_len, d_model]
        pos_encoding = pos_encoding.to(device)
        x += pos_encoding  # [b, inp_seq_len, d_model]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, attn_block1, attn_block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
            # => [b, targ_seq_len, d_model], [b, num_heads, targ_seq_len, targ_seq_len], [b, num_heads, targ_seq_len, inp_seq_len]

            attention_weights[f'decoder_layer{i + 1}_block1'] = attn_block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = attn_block2

        return x, attention_weights
        # => [b, targ_seq_len, d_model],
        # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
        #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}


def create_padding_mask(seq, padding_idx):
    """
    :param seq: [b, seq_len]
    :param padding_idx: 0 or other
    :return:
    """
    if padding_idx == 0:  # 重要
        seq = torch.eq(seq, torch.tensor(0)).float()
    else:
        seq = torch.eq(seq, torch.tensor(padding_idx)).float()
    return seq[:, np.newaxis, np.newaxis, :]  # =>[b, 1, 1, seq_len]


def create_look_ahead_mask(seq_len):
    """
torch.triu(tensor, diagonal=0) 求上三角矩阵，diagonal默认为0表示主对角线的上三角矩阵
diagonal>0，则主对角上面的第|diagonal|条次对角线的上三角矩阵
diagonal<0，则主对角下面的第|diagonal|条次对角线的上三角矩阵
    :param seq_len:
    :return:
    """
    mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)
    # mask = mask.device() #
    return mask  # [seq_len, seq_len]


# inp [b, inp_seq_len] 序列已经加入pad填充
# targ [b, targ_seq_len] 序列已经加入pad填充
def create_mask(inp, targ, padding_idx, device):
    # encoder padding mask
    enc_padding_mask = create_padding_mask(inp, padding_idx)  # =>[b,1,1,inp_seq_len] mask=1的位置为pad

    # decoder's first attention block(self-attention)
    # 使用的padding create_mask & look-ahead create_mask
    look_ahead_mask = create_look_ahead_mask(targ.shape[-1])  # =>[targ_seq_len,targ_seq_len] ##################
    look_ahead_mask = look_ahead_mask.to(device)
    dec_targ_padding_mask = create_padding_mask(targ, padding_idx)  # =>[b,1,1,targ_seq_len]
    dec_targ_padding_mask = dec_targ_padding_mask.to(device)
    combined_mask = torch.max(look_ahead_mask, dec_targ_padding_mask)  # 結合了2 種mask => [b,1,targ_seq_len,targ_seq_len]

    # decoder's second attention block(encoder-decoder attention) 使用的padding create_mask
    # 【注意】：这里的mask是用于遮挡encoder output的填充pad，而encoder的输出与其输入shape都是[b,inp_seq_len,d_model]
    # 所以这里mask的长度是inp_seq_len而不是targ_mask_len
    dec_padding_mask = create_padding_mask(inp, padding_idx)  # =>[b,1,1,inp_seq_len] mask=1的位置为pad

    return enc_padding_mask, combined_mask, dec_padding_mask
    # [b,1,1,inp_seq_len], [b,1,targ_seq_len,targ_seq_len], [b,1,1,inp_seq_len]


class Transformer(torch.nn.Module):
    def __init__(self,
                 num_layers,  # N个encoder layer
                 d_model,
                 num_heads,
                 dff,  # 点式前馈网络内层fn的维度
                 input_vocab_size,  # input 詞表大小
                 target_vocab_size,  # target 詞表大小
                 pe_input,  # input max_pos_encoding
                 pe_target,  # input max_pos_encoding
                 rate=0.1):
        super(Transformer, self).__init__()
        self.padding_idx = 0
        self.encoder = Encoder(num_layers,
                               d_model,
                               num_heads,
                               dff,
                               input_vocab_size,
                               pe_input,
                               rate)
        self.decoder = Decoder(num_layers,
                               d_model,
                               num_heads,
                               dff,
                               target_vocab_size,
                               pe_target,
                               rate)
        self.final_layer = torch.nn.Linear(d_model, target_vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    # inp [b, inp_seq_len]
    # targ [b, targ_seq_len]
    # enc_padding_mask [b, 1, 1, inp_seq_len]
    # look_ahead_mask [b, 1, targ_seq_len, targ_seq_len]
    # dec_padding_mask [b, 1, 1, inp_seq_len] # 注意这里的维度是inp_seq_len
    def forward(self, inp, targ, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, enc_padding_mask)  # =>[b, inp_seq_len, d_model]

        dec_output, attention_weights = self.decoder(targ, enc_output, look_ahead_mask, dec_padding_mask)
        # => [b, targ_seq_len, d_model],
        # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
        #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}
        final_output = self.final_layer(dec_output)  # =>[b, targ_seq_len, target_vocab_size]

        return final_output, attention_weights
        # [b, targ_seq_len, target_vocab_size]
        # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
        #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}

    def train_step(self, inp, targ):
        device = next(self.parameters()).device
        # 目标（target）被分成了 tar_inp 和 tar_real
        # tar_inp 作为输入传递到解码器。
        # tar_real 是位移了 1 的同一个输入：在 tar_inp 中的每个位置，tar_real 包含了应该被预测到的下一个标记（token）。
        inp = inp.to(device)
        targ = targ.to(device)
        targ_inp = targ[:, :-1]
        targ_real = targ[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_mask(inp, targ_inp, self.padding_idx, device)

        inp = inp.to(device)
        targ_inp = targ_inp.to(device)
        targ_real = targ_real.to(device)
        enc_padding_mask = enc_padding_mask.to(device)
        combined_mask = combined_mask.to(device)
        dec_padding_mask = dec_padding_mask.to(device)
        # print('device:', inp.device, targ_inp)

        # forward
        prediction, _ = self(inp, targ_inp, enc_padding_mask, combined_mask, dec_padding_mask)
        # [b, targ_seq_len, target_vocab_size]
        # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
        #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}

        #loss = mask_loss_func(targ_real, prediction)
        #metric = mask_accuracy_func(targ_real, prediction)
        return prediction, targ_real

    def calculate_loss(self, x_hat, y):
        # x_hat = x_hat.contiguous().view(-1, x_hat.size(-1))
        # y = y.contiguous().view(-1)
        # device = next(self.parameters()).device
        # x_hat = x_hat.permute(0, 2, 1).to(device)
        pad = 0
        x_hat = x_hat.transpose(-1, -2)
        loss = self.loss_fn(x_hat, y)
        # mask = torch.logical_not(y.eq(pad)).type(loss.dtype)
        # info(f" {mask.shape=}")
        # loss *= mask
        # return loss.sum() / mask.sum().item()
        return loss

    def infer(self, vocab, inp_sentence, max_length=None):
        device = next(self.parameters()).device
        inp_sentence_ids = vocab.encode_to_tokens(inp_sentence)
        encoder_input = torch.tensor(inp_sentence_ids).unsqueeze(dim=0)  # =>[b=1, inp_seq_len]

        decoder_input = [vocab.bos_idx]
        decoder_input = torch.tensor(decoder_input).unsqueeze(0)  # =>[b=1,seq_len=1]

        max_length = (len(inp_sentence_ids) + 2) if max_length is None else max_length
        with torch.no_grad():
            for i in range(max_length):
                enc_padding_mask, combined_mask, dec_padding_mask = create_mask(encoder_input,
                                                                                decoder_input,
                                                                                padding_idx=self.padding_idx,
                                                                                device=device)
                # [b,1,1,inp_seq_len], [b,1,targ_seq_len,inp_seq_len], [b,1,1,inp_seq_len]

                encoder_input = encoder_input.to(device)
                decoder_input = decoder_input.to(device)
                enc_padding_mask = enc_padding_mask.to(device)
                combined_mask = combined_mask.to(device)
                dec_padding_mask = dec_padding_mask.to(device)

                # forward
                predictions, attention_weights = self(encoder_input,
                                                      decoder_input,
                                                      enc_padding_mask,
                                                      combined_mask,
                                                      dec_padding_mask)
                # [b=1, targ_seq_len, target_vocab_size]
                # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
                #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}

                # 看最後一個詞並計算它的 argmax
                prediction = predictions[:, -1:, :]  # =>[b=1, 1, target_vocab_size]
                prediction_id = torch.argmax(prediction, dim=-1)  # => [b=1, 1]
                # print('prediction_id:', prediction_id, prediction_id.dtype) # torch.int64
                if prediction_id.squeeze().item() == vocab.eos_idx:
                    # return decoder_input.squeeze(dim=0), attention_weights
                    break

                # 連接decoder_input
                decoder_input = torch.cat([decoder_input, prediction_id],
                                          dim=-1)  # [b=1,targ_seq_len=1]=>[b=1,targ_seq_len=2]

        # return decoder_input.squeeze(dim=0), attention_weights
        return decoder_input.squeeze(dim=0)


if __name__ == '__main__':
    device = 'cpu'
    sample_transformer = Transformer(num_layers=2,
                                     d_model=512,
                                     num_heads=8,
                                     dff=2048,
                                     input_vocab_size=8500,
                                     target_vocab_size=8000,
                                     pe_input=10000,
                                     pe_target=6000)
    sample_transformer = sample_transformer.to(device)

    temp_inp = torch.tensor(np.random.randint(low=0, high=200, size=(64, 42)))  # [b, inp_seq_len]
    temp_targ = torch.tensor(np.random.randint(low=0, high=200, size=(64, 36)))  # [b, targ_seq_len]

    fn_out, attn = sample_transformer(temp_inp.cuda(), temp_targ.cuda(), None, None, None)
    print(fn_out.shape)  # [64, 36, 8000]
    print(attn['decoder_layer2_block1'].shape)  # [64, 8, 36, 36]
    print(attn['decoder_layer2_block2'].shape)  # [64, 8, 36, 42]
