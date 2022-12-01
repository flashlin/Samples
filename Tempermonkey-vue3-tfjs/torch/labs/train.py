import torch
from torch import nn, Tensor

from preprocess_data import Seq2SeqDataset, convert_translation_file_to_csv
from ml.lit import BaseLightning, start_train, PositionalEncoding, load_model
from utils.linq_tokenizr import LINQ_VOCAB_SIZE, linq_encode, linq_decode
from utils.tokenizr import PAD_TOKEN_VALUE, BOS_TOKEN_VALUE, EOS_TOKEN_VALUE
from utils.tsql_tokenizr import TSQL_VOCAB_SIZE, tsql_decode


class SeqEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model)

    def forward(self, x):
        """
        :param x: [long]
        :return: [batch, seq_len, dim]
        """
        output = self.embedding(x)
        output = self.positional(output)
        return output

class TranslateModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_dim=3, num_heads=8):
        super().__init__()
        d_model = d_dim * num_heads
        self.src_embedding = SeqEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = SeqEmbedding(tgt_vocab_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.predictor = nn.Linear(in_features=d_model, out_features=tgt_vocab_size)

    def forward(self, batch):
        src, tgt = batch
        src = src.unsqueeze(0)
        tgt = tgt.unsqueeze(0)
        x_embedding = self.src_embedding(src)
        y_embedding = self.tgt_embedding(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(-1))
        src_key_padding_mask = TranslateModel.get_key_padding_mask(src)
        tgt_key_padding_mask = TranslateModel.get_key_padding_mask(tgt)
        output = self.transformer(x_embedding, y_embedding,
                              tgt_mask=tgt_mask,
                              src_key_padding_mask=src_key_padding_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask)
        # output = [batch, seq_len, word_dim]
        logits = self.proj_vocab_layer(output)
        logits = torch.einsum('ijk->jik', logits)
        return logits, y_embedding

    @staticmethod
    def get_key_padding_mask(tokens):
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == PAD_TOKEN_VALUE] = -torch.inf
        return key_padding_mask

def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e.
              the length of the input sequence to the model),
              and for tgt masking, this must be target sequence length
    Return:
        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


class LitTranslator(BaseLightning):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.model = TranslateModel(src_vocab_size, tgt_vocab_size)
        self.criterion = nn.CrossEntropyLoss() #reduction="none")
        self.init_dataloader(Seq2SeqDataset("../output/linq-sample.csv"), 2)

    def _fetch_xy_batch(self, batch):
        x, y, x_lens, y_lens = batch
        return x, y

    def _calculate_loss(self, batch, mode="train"):
        x, y = batch
        x_hat, y_true = x
        loss = self.criterion(x_hat, y_true)
        self.log("%s_loss" % mode, loss)
        return loss

    def _calculate_loss1(self, y_pred):
        y_pred_copy = y_pred.detach()
        dec_output_copy = dec_output.detach()

        # loss 계산을 위해 shape 변경
        y_pred = y_pred.reshape(-1, y_pred.size(-1))
        dec_output = dec_output.view(-1).long()

        # padding 제외한 value index 추출
        real_value_index = [dec_output != 0]

        # padding은 loss 계산시 제외
        mb_loss = self.criterion(y_pred[real_value_index], dec_output[real_value_index])  # Input: (N, C) Target: (N)
        return mb_loss

    def evaluate(self, input_text, max_length=300):
        enc_input = torch.tensor(linq_encode(input_text))
        dec_input = torch.tensor([[BOS_TOKEN_VALUE]])
        device = 'cpu'
        for i in range(max_length):
            batch = enc_input.to(device), dec_input.to(device)
            y_pred = self.model(batch)
            y_pred_ids = y_pred.max(dim=-1)[1]
            if (y_pred_ids[0, -1] == EOS_TOKEN_VALUE).to(torch.device('cpu')).numpy():
                return LitTranslator.decoding_from_result(enc_input=enc_input, y_pred=y_pred)

            # decoding_from_result(enc_input, y_pred, tokenizer)
            dec_input = torch.cat([dec_input.to(torch.device('cpu')),
                                   y_pred_ids[0, -1].unsqueeze(0).unsqueeze(0).to(torch.device('cpu'))], dim=-1)

            if i == max_length - 1:
                return LitTranslator.decoding_from_result(enc_input=enc_input, y_pred=y_pred)


    @staticmethod
    def decoding_from_result(enc_input, y_pred):
        list_of_input_ids = enc_input.tolist()
        list_of_pred_ids = y_pred.max(dim=-1)[1].tolist()
        input_token = linq_decode(list_of_input_ids)
        pred_token = tsql_decode(list_of_pred_ids)
        return pred_token



def prepare_train_data():
    print("convert translation file to csv...")
    convert_translation_file_to_csv()
    print("done.")

def main():
    start_train(LitTranslator, device='cpu',
                max_epochs=100,
                src_vocab_size=LINQ_VOCAB_SIZE,
                tgt_vocab_size=TSQL_VOCAB_SIZE)

def infer():
    model = load_model(LitTranslator, src_vocab_size=LINQ_VOCAB_SIZE, tgt_vocab_size=TSQL_VOCAB_SIZE)
    sql = model.evaluate('from tb3 in customer select new tb3')
    print(sql)

if __name__ == "__main__":
    #prepare_train_data()
    #main()
    infer()