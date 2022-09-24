import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

from Seq2SeqTransformer import Seq2SeqTransformer
# from Seq2SeqTransformer import Seq2SeqTransformer
from data_utils import load_csv_to_dataframe, split_dataframe

# import warnings
# warnings.filterwarnings("ignore")

# data manupulation libs

from train_utils import build_vocab_from_dataframe, generate_batch, train_epoch, evaluate

# from pandarallel import pandarallel
# Initialization
# pandarallel.initialize()


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    data_df = load_csv_to_dataframe()
    train_df, val_df = split_dataframe(data_df)

    # create a vocab class with freq_threshold=0 and max_size=100
    # voc = Vocabulary(0, 100)
    # sentence_list = ['that is a cat', 'that is not a dog']
    # # build vocab
    # voc.build_vocabulary(sentence_list)
    #
    # print('index to string: ', voc.itos)
    # print('string to index:', voc.stoi)
    # print('numericalize -> cat and a dog: ', voc.numericalize('cat and a dog'))
    #
    # ####
    # train_dataset = Train_Dataset(train_df, 'source_sentence', 'target_sentence')
    # print(train_df.loc[1])
    # print(train_dataset[1])
    #
    # ###
    # train_loader = get_train_loader(train_dataset, 32)
    # source = next(iter(train_loader))[0]
    # target = next(iter(train_loader))[1]
    #
    # print('source: \n', source)
    #
    # print('source shape: ', source.shape)
    # print('target shape: ', target.shape)

    source_column_name = "source_sentence"
    target_column_name = "target_sentence"

    source_tokenizer = get_tokenizer('basic_english')
    target_tokenizer = get_tokenizer('basic_english')

    txt = target_tokenizer("All residents aged 20 to 59 years who live in Japan must enroll in public pension system.")
    print(f"{txt}")

    source_vocab = build_vocab_from_dataframe(train_df, source_column_name, source_tokenizer)
    target_vocab = build_vocab_from_dataframe(train_df, target_column_name, target_tokenizer)

    #
    # train_data_file = "./input_data/linq_corpus.csv"
    #
    SRC_VOCAB_SIZE = len(source_vocab)
    TGT_VOCAB_SIZE = len(target_vocab)
    print(f"{SRC_VOCAB_SIZE=}")
    print(f"{TGT_VOCAB_SIZE=}")

    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    NUM_EPOCHS = 20
    PATH = "./checkpoints"
    #
    # BATCH_SIZE = 128
    PAD_IDX = source_vocab['<pad>']
    BOS_IDX = source_vocab['<bos>']
    EOS_IDX = source_vocab['<eos>']
    #
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                     EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                     FFN_HID_DIM)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # train_data = textdata_to_train_data(train_data_file, source_vocab, source_tokenizer,
    #                                     target_vocab, target_tokenizer)
    #
    # # val_data = data_process(val_filepaths)
    # # test_data = data_process(test_filepaths)
    # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    val_data = train_df

    train_iter = DataLoader(train_df, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=generate_batch)
    valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=generate_batch)
    # # test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
    # #     shuffle=True, collate_fn=generate_batch)

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        train_loss = train_epoch(transformer, loss_fn, train_iter, optimizer, DEVICE)
        end_time = time.time()
        val_loss = evaluate(transformer, loss_fn, valid_iter, PAD_IDX)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
               f"Epoch time = {(end_time - start_time):.3f}s"))

    #
    # def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    #     src = src.to(device)
    #     src_mask = src_mask.to(device)
    #     memory = model.encode(src, src_mask)
    #     ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    #     for i in range(max_len - 1):
    #         memory = memory.to(device)
    #         memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
    #         tgt_mask = (generate_square_subsequent_mask(ys.size(0))
    #                     .type(torch.bool)).to(device)
    #         out = model.decode(ys, memory, tgt_mask)
    #         out = out.transpose(0, 1)
    #         prob = model.generator(out[:, -1])
    #         _, next_word = torch.max(prob, dim=1)
    #         next_word = next_word.item()
    #
    #         ys = torch.cat([ys,
    #                         torch.ones(1, 1).type_as(src.data_df).fill_(next_word)], dim=0)
    #         if next_word == EOS_IDX:
    #             break
    #     return ys
    #
    #
    # def translate(model, src, src_vocab, tgt_vocab, src_tokenizer):
    #     model.eval()
    #     tokens = [BOS_IDX] + [src_vocab.stoi[tok] for tok in src_tokenizer(src)] + [EOS_IDX]
    #     num_tokens = len(tokens)
    #     src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
    #     src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    #     tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX, device=DEVICE) \
    #         .flatten()
    #     return " ".join([tgt_vocab.itos[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")
    #
    #
    # output = translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu .", source_vocab, target_vocab,
    #                    source_tokenizer)
    # print(output)
