import numpy as np
import torch
from torch import nn

from common.io import info
from preprocess_data import convert_translation_file_to_csv, TranslationDataset
from ml.lit import BaseLightning, start_train, PositionalEncoding, load_model
from utils.linq_tokenizr import LINQ_VOCAB_SIZE, linq_encode
from utils.tokenizr import PAD_TOKEN_VALUE, BOS_TOKEN_VALUE, EOS_TOKEN_VALUE
from utils.tsql_tokenizr import TSQL_VOCAB_SIZE, tsql_decode

# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention


def get_attn_pad_mask(seq_q, seq_k):
    """
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.words.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).to(seq.device).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)).to(self.device) / np.sqrt(
            d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V).to(self.device)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        device = next(self.parameters()).device
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k) \
            .transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k) \
            .transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v) \
            .transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1) \
            .repeat(1, n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(device)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        device = next(self.parameters()).device
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_TOKEN_VALUE)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size):
        super().__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_TOKEN_VALUE)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        """
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0)  # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, d_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_dim)
        self.positional_encoding = PositionalEncoding(d_model=d_dim, dropout=0)

    def forward(self, x):
        output = self.embedding(x).transpose(0, 1)
        output = self.positional_encoding(output).transpose(0, 1)
        return output


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_dim):
        super().__init__()
        self.src_emb = WordEmbedding(src_vocab_size, d_dim)
        self.tgt_emb = WordEmbedding(tgt_vocab_size, d_dim)
        self.transformer = nn.Transformer(d_model=d_dim,
                                          num_encoder_layers=7,
                                          num_decoder_layers=7,
                                          dim_feedforward=d_model,
                                          batch_first=True)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, src_inputs, tgt_inputs):
        src = self.src_emb(src_inputs)
        tgt = self.tgt_emb(tgt_inputs)
        outputs, _ = self.transformer(src, tgt)
        dec_logits = self.projection(outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits
        # logits = dec_logits.view(-1, dec_logits.size(-1))
        # return logits

    def inference(self, input_text):
        device = next(self.parameters()).device
        enc_inputs = torch.tensor([linq_encode(input_text)]).to(device)
        greedy_dec_input = self.greedy_decoder(enc_inputs, start_symbol=BOS_TOKEN_VALUE)
        predict, _, _, _ = self(enc_inputs, greedy_dec_input)
        predict = predict.words.max(1, keepdim=False)[1]
        return predict.tolist()

    def greedy_decoder(self, enc_input, start_symbol):
        """
        :param model: Transformer Model
        :param enc_input: The encoder input
        :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
        :return: The target input
        """
        enc_input = self.src_emb(enc_input)
        enc_outputs, enc_self_attns = self.transformer.encoder(enc_input)
        dec_input = torch.zeros(1, 0).type_as(enc_input.words)
        terminal = False
        next_symbol = start_symbol
        while not terminal:
            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(enc_input.device)], -1)
            dec_outputs, _, _ = self.transformer.decoder(dec_input, enc_input, enc_outputs)
            projected = self.projection(dec_outputs)
            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
            # print(f" {prob.data=}")
            next_word = prob.words[-1]  # 拿最後一個字
            next_symbol = next_word
            if next_symbol == EOS_TOKEN_VALUE:
                terminal = True
        return dec_input

    @staticmethod
    def get_key_padding_mask(tokens):
        key_padding_mask = torch.zeros(tokens.size()).to(tokens.device)
        key_padding_mask[tokens == PAD_TOKEN_VALUE] = - torch.inf
        return key_padding_mask


class LitTranslator(BaseLightning):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.model = Seq2SeqTransformer(src_vocab_size, tgt_vocab_size, d_dim=512)
        self.criterion = nn.CrossEntropyLoss()  # reduction="none")
        self.init_dataloader(TranslationDataset("../output/linq-sample.csv"), 2)

    def forward(self, batch):
        enc_inputs, dec_inputs, dec_outputs = batch
        info(f" f {enc_inputs.shape=}")

        logits = self.model(enc_inputs, dec_inputs)
        info(f" f {logits.shape=}")
        return logits, dec_outputs

    def _calculate_loss(self, data, mode="train"):
        (logits, dec_outputs), batch = data
        info(f" {logits.shape=} {dec_outputs.shape=}")
        loss = self.criterion(logits, dec_outputs.view(-1))
        self.log("%s_loss" % mode, loss)
        return loss

    def infer(self, text):
        sql_values = self.model.inference(text)
        sql = tsql_decode(sql_values)
        return sql


def prepare_train_data():
    print("convert translation file to csv...")
    convert_translation_file_to_csv()
    print("done.")


def main():
    start_train(LitTranslator, device='cuda',
                max_epochs=100,
                src_vocab_size=LINQ_VOCAB_SIZE,
                tgt_vocab_size=TSQL_VOCAB_SIZE,
                model_name='Task13')


def infer():
    model = load_model(LitTranslator, src_vocab_size=LINQ_VOCAB_SIZE, tgt_vocab_size=TSQL_VOCAB_SIZE,
                       model_name='Task13')
    text = 'from tb3 in customer select new tb3'
    print("start infer")
    sql = model.infer(text)
    print(text)
    print(sql)


if __name__ == "__main__":
    info(f" {LINQ_VOCAB_SIZE=} {TSQL_VOCAB_SIZE=} {PAD_TOKEN_VALUE=}")
    # prepare_train_data()
    main()
    # infer()
