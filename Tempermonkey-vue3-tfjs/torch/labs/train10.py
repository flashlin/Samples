import torch
from pytorch_lightning import Trainer
from torch import nn

from common.io import info
from ml.lit import BaseLightning, start_train, PositionalEncoding, load_model
from prepare7 import Linq2TSqlDataset
from utils.linq_tokenizr import LINQ_VOCAB_SIZE, linq_encode
from utils.tsql_tokenizr import TSQL_VOCAB_SIZE

# inp: 依照一個句子,
# out: 預測最後一個字
# .unsqueeze
# .squeeze
# .transpose(1, 2) 交換維度

# https://blog.51cto.com/u_11466419/5530949
def padding_tensor_sequence(sequence, padding_value, max_length):
    seq_len = len(sequence)
    target = torch.zeros(max_length, dtype=torch.long).fill_(padding_value)
    target[:seq_len] = sequence
    return target
    # return F.pad(input=sequence, pad=(0,1), mode='constant', value=padding_value)

def padding_batch(sequences, padding_value):
    num = len(sequences)
    max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len)
    out_tensor = sequences[0].words.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
    return out_tensor
    # #lp = torch.stack([torch.cat([i, i.new_zeros(emb_len - i.size(0))], 0) for i in l], 1)
    # out = torch.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=padding_value)
    # return out.transpose(1, 2)

def generate_tgt(batch, padding_idx):
    x, y = batch
    max_len = len(x)
    # tgt 不要最後一個token
    # tgt = x[:, :-1]
    tgt = x[:-1]
    tgt = padding_tensor_sequence(tgt, padding_idx, max_len)
    # tgt_y 不要第一個的token
    # tgt_y = x[:, 1:]
    tgt_y = x[1:] #F.pad(x[1:], [padding_idx] * 2)
    tgt_y = padding_tensor_sequence(tgt_y, padding_idx, max_len)
    # 計算即要預測的有效 token 的數量, 後面計算 loss 要用
    n_tokens = (tgt_y != padding_idx).sum()
    return x, tgt, tgt_y, n_tokens

# github.com/graykode/nlp-tutorial/tree/master
class Translator(nn.Module):
    def __init__(self, tgt_vocab_size, tgt_seq_len, tgt_dim=128):
        super().__init__()
        self.padding_index = 3
        self.embedding = nn.Embedding(num_embeddings=tgt_vocab_size, embedding_dim=tgt_dim)
        self.transformer = nn.Transformer(d_model=128, num_encoder_layers=2,
                                          num_decoder_layers=2, dim_feedforward=512,
                                          batch_first=True)
        self.positional_encoding = PositionalEncoding(d_model=128, dropout=0)
        self.predictor = nn.Linear(tgt_dim, tgt_seq_len)

    def forward(self, src, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[- 1])
        src_key_padding_mask = Translator.get_key_padding_mask(src, self.padding_index).unsqueeze(0)
        tgt_key_padding_mask = Translator.get_key_padding_mask(tgt, self.padding_index).unsqueeze(0)
        src = src.unsqueeze(0)
        tgt = tgt.unsqueeze(0)
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return out

    @staticmethod
    def get_key_padding_mask(tokens, padding_index):
        """
        用於key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == padding_index] = -torch.inf
        return key_padding_mask


class LitTranslator(BaseLightning):
    def __init__(self, tgt_len, tgt_vocab_size):
        super().__init__()
        self.loss = None
        self.model = Translator(tgt_vocab_size, tgt_seq_len=tgt_len)
        self.criteria = nn.CrossEntropyLoss()
        self.init_dataloader(Linq2TSqlDataset('../output/linq-sample.csv'), 2)

    def forward(self, batch):
        src, tgt, tgt_y, n_tokens = generate_tgt(batch, padding_idx=3)
        out = self.model(src, tgt)
        self.loss = self.criteria(out.contiguous().view(- 1, out.size(- 1)), tgt_y.contiguous().view(- 1)) / n_tokens
        return out

    # def training_step(self, batch, batch_idx):
    #     self.loss = self.criteria(out.contiguous().view(- 1, out.size(- 1)), tgt_y.contiguous().view(- 1)) / n_tokens
    #     return out

    def _calculate_loss(self, batch, mode="train"):
        self.log("%s_loss" % mode, self.loss)
        return self.loss

    def infer(self, src, max_length):
        src = torch.tensor(src, dtype=torch.long)
        model = self.model.eval()
        tgt = torch.tensor([0], dtype=torch.long)
        for i in range(max_length):
            out = model(src, tgt)
            # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
            #predict = model.predictor(out[:, -1])
            predict = model.predictor(out[-1])
            # predict = model(out[-1], tgt)
            # 找出最大值的index
            y = torch.argmax(predict, dim=1)
            # 和之前的预测结果拼接到一起
            #tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
            tgt = torch.concat([tgt, y], dim=0)
            # 如果為 <eos> 說明預測結束，跳出循環
            if y[-1] == 2:
                break
        print(tgt)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # start_train(LitSeq2Seq, device='cpu', src_vocab_size=LINQ_VOCAB_SIZE, tgt_vocab_size=TSQL_VOCAB_SIZE)
    info(f" {LINQ_VOCAB_SIZE=} {TSQL_VOCAB_SIZE=}")
    start_train(LitTranslator, device='cpu',
                max_epochs=100,
                #src_len=277,
                #src_vocab_size = LINQ_VOCAB_SIZE,
                tgt_len=100,
                tgt_vocab_size = TSQL_VOCAB_SIZE)

def test():
    s1 = "from tb3 in account select tb3.name"
    v1 = linq_encode(s1)
    model = load_model(LitTranslator)
    v2 = model.infer(v1, 200)
    print(f" {v2=}")

def test2():
    s1 = "from tb3 in account select tb3.name"
    v1 = linq_encode(s1)
    trainer = Trainer()
    model = load_model(LitTranslator)
    predictions = trainer.predict(model, enumerate([v1]))
    print(f" {predictions=}")

if __name__ == "__main__":
    #main()
    test()