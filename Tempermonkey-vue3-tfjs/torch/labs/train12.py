from torch import nn

from common.io import info
from ml.seq2seq_model import Seq2SeqTransformer
from preprocess_data import convert_translation_file_to_csv, TranslationDataset
from ml.lit import BaseLightning, start_train, load_model
from utils.linq_tokenizr import LINQ_VOCAB_SIZE
from utils.tokenizr import PAD_TOKEN_VALUE
from utils.tsql_tokenizr import TSQL_VOCAB_SIZE, tsql_decode

# worked but 會隨機翻譯

# Transformer Parameters


class LitTranslator(BaseLightning):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.model = Seq2SeqTransformer(src_vocab_size, tgt_vocab_size)
        self.criterion = nn.CrossEntropyLoss()  # reduction="none")
        self.init_dataloader(TranslationDataset("../output/linq-sample.csv"), 1)

    def forward(self, batch):
        enc_inputs, dec_inputs, dec_outputs = batch
        logits, enc_self_attns, dec_self_attns, dec_enc_attns = self.model(enc_inputs, dec_inputs)
        return logits, dec_outputs

    def _calculate_loss(self, data, mode="train"):
        (logits, dec_outputs), batch = data
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
                tgt_vocab_size=TSQL_VOCAB_SIZE)


def infer():
    model = load_model(LitTranslator, src_vocab_size=LINQ_VOCAB_SIZE, tgt_vocab_size=TSQL_VOCAB_SIZE)
    print("start infer")

    def inference(text):
        print(text)
        sql = model.infer(text)
        print(sql)

    inference('from tb3 in customer select new tb3')
    inference('from c in customer select new { c.id, c.name }')


if __name__ == "__main__":
    info(f" {LINQ_VOCAB_SIZE=} {TSQL_VOCAB_SIZE=} {PAD_TOKEN_VALUE=}")
    #convert_translation_file_to_csv()
    #info("prepare train data")
    #prepare_train_data()
    #info("start train")
    #main()
    infer()
