import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader

from common.io import info, remove_file
from ml.data_utils import get_data_file_path
from ml.lit import BaseLightning, start_train, copy_last_ckpt, load_model, PositionalEncoding
from labs.my_model import encode_src_tokens, src_char2index, src_symbols, \
    tgt_char2index, \
    decode_src_to_text, line_to_tokens, read_examples
from utils.data_utils import df_intstr_to_values, pad_array
from utils.stream import int_list_to_str


def pad_row_iter(row, max_seq_len, padding_idx):
    src, tgt1, tgt2 = row
    items_lens = [len(src), len(tgt1), len(tgt2)]
    is_all_same_len = all(x == [0] for x in items_lens)
    src = pad_array(src, padding_idx, max_seq_len)
    tgt1 = pad_array(tgt1, padding_idx, max_seq_len)
    tgt2 = pad_array(tgt2, padding_idx, max_seq_len)
    if is_all_same_len:
        yield src, tgt1, tgt2
        return
    max_len = max(items_lens)
    for n in range(max_len):
        new_src = pad_array(src[n: n + max_len], padding_idx, max_seq_len)
        new_tgt1 = pad_array(tgt1[n: n + max_len], padding_idx, max_seq_len)
        new_tgt2 = pad_array(tgt2[n: n + max_len], padding_idx, max_seq_len)
        yield new_src, new_tgt1, new_tgt2


def read_examples_to_tokens(example_file, num_tuple):
    for idx, tokens in enumerate(read_examples(example_file)):
        tuple_value = []
        for n in range(num_tuple):
            tuple_value.append(tokens)
        yield tuple(tuple_value)


def write_train_files(max_seq_len, target_path="./output"):
    def write_train_data(a_tuple):
        a_list = list(a_tuple)
        for idx, column in enumerate(a_list):
            f.write(int_list_to_str(column))
            if idx < len(a_list) - 1:
                f.write('\t')
        f.write('\n')

    target_csv_file = f"{target_path}\\linq_sql.csv"
    remove_file(target_csv_file)
    example_file = get_data_file_path("linq_classification1.txt")
    with open(target_csv_file, "w", encoding='UTF-8') as f:
        f.write("src\ttgt\n")
        for (src, tgt) in read_examples_to_tokens(example_file, 2):
            row = encode_src_tokens(src), encode_src_tokens(tgt)
            write_train_data(row)
            # for row_tuple in pad_row_iter(row, max_seq_len, src_char2index['<pad>']):
            #     new_src, new_tgt1, new_tgt2 = row_tuple
            #     assert len(new_src) == len(new_tgt1)
            #     assert len(new_src) == len(new_tgt2)
            #     write_train_data(row_tuple)


def pad_data_loader(dataset, batch_size, padding_idx, **kwargs):
    def pad_collate(batch):
        a_tuple = zip(*batch)
        result = []
        for column in list(a_tuple):
            result.append(torch.nn.utils.rnn.pad_sequence(column, batch_first=True, padding_value=padding_idx))
        return tuple(result)

    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=pad_collate, **kwargs)


class TranslationDataset(Dataset):
    def __init__(self, csv_file_path, padding_idx):
        self.padding_idx = padding_idx
        self.df = df = pd.read_csv(csv_file_path, sep='\t')
        self.src = df_intstr_to_values(df['src'])
        self.tgt = df_intstr_to_values(df['tgt'])

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        tgt = self.tgt[idx]
        # enc_input = torch.tensor(pad_array(src[1:-1], self.padding_idx, max_len), dtype=torch.long)
        # dec_input = torch.tensor(pad_array(tgt[:-1], self.padding_idx, max_len), dtype=torch.long)
        # dec_output = torch.tensor(pad_array(tgt[1:], self.padding_idx, max_len), dtype=torch.long)
        src = torch.tensor(src, dtype=torch.long)
        tgt = torch.tensor(tgt, dtype=torch.long)
        return src, tgt

    def create_dataloader(self, batch_size=32):
        train_size = int(0.8 * len(self))
        val_size = len(self) - train_size
        train_data, val_data = random_split(self, [train_size, val_size])
        train_loader = pad_data_loader(train_data, batch_size=batch_size, padding_idx=self.padding_idx)
        val_loader = pad_data_loader(val_data, batch_size=batch_size, padding_idx=self.padding_idx)
        return train_loader, val_loader


PADDING_IDX = src_char2index['<pad>']


def get_key_padding_mask(tokens):
    key_padding_mask = torch.zeros(tokens.size()).type(torch.bool)
    # key_padding_mask[tokens == PADDING_IDX] = -torch.inf
    key_padding_mask = (tokens == PADDING_IDX)
    return key_padding_mask


class TransformerTagger(nn.Module):
    def __init__(self, src_vocab_size, embedding_dim, hidden_feature_dim, classes_num):
        super().__init__()
        self.input_feature_dim = embedding_dim
        self.hidden_feature_dim = hidden_feature_dim

        self.embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.pos_embedding = PositionalEncoding(d_model=embedding_dim, dropout=0.1)

        self.transformer = nn.Transformer(d_model=embedding_dim,
                                          num_encoder_layers=7,
                                          num_decoder_layers=7,
                                          dim_feedforward=512,
                                          nhead=8,
                                          batch_first=True)

        # self.linear = nn.Linear(hidden_feature_dim, classes_num)
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, classes_num),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.BCELoss()
        # self.loss_fn = nn.NLLLoss()

    def forward(self, x, y):
        device = x.device
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(y.size()[-1]).to(device)
        src_key_padding_mask = get_key_padding_mask(x).to(device)
        tgt_key_padding_mask = get_key_padding_mask(y).to(device)

        x = self.embedding(x)
        x = self.pos_embedding(x)
        y = self.embedding(y)
        y = self.pos_embedding(y)
        output = self.transformer(x, y,
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask
                                  )  # [batch_size, tgt_seq_len, embedding_dim]
        # 在訓練時, 把output 的所有輸出送给Linear
        # 而在推理時, 只需要將最後一個輸出送给Linear 即可, 即 output[:-1]
        # output = reduce_dim(output)
        output = self.linear(output)
        # _, predictive_value = torch.max(output, 1)  # 從output中取最大的出來作為預測值
        # predictive_value = F.log_softmax(output, dim=1)  # 從output中取最大的出來作為預測值
        return output

    def calculate_loss(self, x, y):
        x = x.contiguous().view(-1, x.size(-1))
        y = y.contiguous().view(-1)
        # info(f" loss {x.shape=} {y.shape=}")
        return self.loss_fn(x, y)

    def inference(self, text, max_length):
        self.eval()
        device = next(self.parameters()).device
        text_tokens = line_to_tokens(text)
        text_values = encode_src_tokens(text_tokens)
        src = torch.LongTensor([text_values]).to(device)
        tgt = torch.LongTensor([[tgt_char2index['<bos>']]]).to(device)
        for i in range(max_length):
            out = self(src, tgt)
            info(f" {out.shape=}")
            last_out = out[:, -1]
            info(f" {last_out.shape=}")
            predict = self.linear(last_out)
            # 找出最大值的index
            y = torch.argmax(predict, dim=1)
            # 和之前的预测结果拼接到一起
            tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)

            # 如果为<eos>，说明预测结束，跳出循环
            if y == tgt_char2index['<eos>']:
                break
        # print(tgt)
        return tgt


class MyModel2(BaseLightning):
    def __init__(self):
        super().__init__()
        batch_size = 16
        self.model = TransformerTagger(src_vocab_size=len(src_symbols),
                                       embedding_dim=8,
                                       hidden_feature_dim=len(src_symbols),
                                       classes_num=len(src_symbols),
                                       )
        self.init_dataloader(TranslationDataset("../output/linq_sql.csv", src_char2index['<pad>']), batch_size)

    def forward(self, batch):
        src, tgt = batch
        logits = self.model(src, tgt)
        return logits, tgt

    def _calculate_loss(self, data, mode="train"):
        (logits, tgt), batch = data
        loss = self.model.calculate_loss(logits, tgt)
        self.log("%s_loss" % mode, loss)
        return loss

    def infer(self, text):
        values = self.model.inference(text, 100)
        sql = decode_src_to_text(values)
        return sql


def evaluate():
    print(f"test")
    model = load_model(MyModel2)
    assert model is not None

    def inference(text):
        print(text)
        sql = model.infer(text)
        print(sql)

    inference('from tb3 in customer select new tb3')
    # inference('from c in customer select new { c.id, c.name }')
    # inference('from c in customer join p in products on c.id equals p.id select new { c.id, c.name, p.name }')


def train():
    MAX_SEQ_LEN = 100
    print("prepare train data...")
    write_train_files(max_seq_len=MAX_SEQ_LEN)
    copy_last_ckpt(model_name=MyModel2.__name__)
    print("start training...")
    start_train(MyModel2, device='cuda', max_epochs=10)


def test():
    s1 = 'from tb1     in customer where tb1     . price > 1   select tb1     . name'
    tokens = line_to_tokens(s1)
    print(f" {s1=}")
    print(f" {tokens=}")
    v1 = encode_src_tokens(tokens)
    print(f" {v1=}")
    s2 = decode_src_to_text(v1)
    print(s2)
    print(f" {s2=}")


if __name__ == '__main__':
    """
    2022-11-27 loss 平均 3.37 降不下來
    """
    print(f" {torch.__version__=}")
    # test()
    train()
    # evaluate()
