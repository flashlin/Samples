import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from ml.lit import BaseLightning, start_train
from prepare5 import Linq2TSqlDataset
from utils.linq_tokenizr import LINQ_VOCAB_SIZE
from utils.tsql_tokenizr import TSQL_VOCAB_SIZE


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=512, hidden_dim=3, dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # nn.Embedding 可以幫我們建立好字典中每個字對應的 vector
        self.embeddings = nn.Embedding(src_vocab_size, embedding_dim)
        # LSTM layer，形狀為 (input_size, hidden_size, ...)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout)
        self.hidden2out = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, x):
        # LSTM 接受的 input 形狀為 (timesteps, batch, features)，
        # 即 (seq_length, batch_size, embedding_dim)
        # 所以先把形狀為 (batch_size, seq_length) 的 input 轉置後，
        # 再把每個 value (char index) 轉成 embedding vector
        embeddings = self.embeddings(x.t())
        # LSTM 層的 output (lstm_out) 有每個 timestep 出來的結果
        #（也就是每個字進去都會輸出一個 hidden state）
        # 這邊我們取最後一層的結果，即最近一次的結果，來預測下一個字
        lstm_out, _ = self.lstm(embeddings)
        ht = lstm_out[-1]
        # 線性轉換至 output
        out = self.hidden2out(ht)
        return out


class LitSeq2Seq(BaseLightning):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=512, hidden_dim=3, dropout=0.2):
        super().__init__()
        self.model = Seq2Seq(src_vocab_size, tgt_vocab_size, embedding_dim, hidden_dim, dropout)
        ds = Linq2TSqlDataset('../output/linq-sample.csv')
        train_loader, val_loader = ds.create_dataloader()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = val_loader

    def _calculate_loss(self, batch, mode="train"):
        y_hat, y = batch
        loss = F.cross_entropy(y_hat, y)
        acc = self._compute_accuracy(y_hat, y)
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return loss

    def _compute_accuracy(self, y_hat, y):
        return accuracy(y_hat, y)

    def infer(self):
        # pred = model(seq_in)
        pred = 1.0
        # pred = to_prob(F.softmax(pred).data[0].numpy())  # softmax 後轉成機率分佈
        # char = np.random.choice(chars, p=pred)  # 依機率分佈選字
        pass


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_train(LitSeq2Seq, device='cpu', src_vocab_size=LINQ_VOCAB_SIZE, tgt_vocab_size=TSQL_VOCAB_SIZE)

if __name__ == "__main__":
    main()