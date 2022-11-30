from torch import nn


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, padding_idx, word_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, word_dim)
        self.transformer = nn.Transformer(d_model=word_dim, batch_first=True)
        self.predictor = nn.Linear(word_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=padding_idx)

    def forward(self, x, y):
        outputs = self.transform(x, y)
        outputs = self.predictor(outputs)
        return outputs

    def transform(self, x, y):
        src_key_padding_mask = self.get_key_padding_mask(x)
        tgt_key_padding_mask = self.get_key_padding_mask(y)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(y.size(-1)).to(x.device)
        x = self.embedding(x)
        y = self.embedding(y)
        outputs = self.transformer(x, y,
                                   tgt_mask=tgt_mask,
                                   src_key_padding_mask=src_key_padding_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask
                                   )
        return outputs

    def calculate_loss(self, x_hat, y):
        n_tokens = (y != self.padding_idx).sum()

        x_hat = x_hat.contiguous().view(-1, x_hat.size(-1))
        y = y.contiguous().view(-1)

        loss = self.loss_fn(x_hat, y) / n_tokens
        return loss

    def get_key_padding_mask(self, tokens):
        key_padding_mask = tokens == self.padding_idx
        # # key_padding_mask = self.transformer.generate_square_subsequent_mask(tokens.size())
        # key_padding_mask = torch.zeros(tokens.size()).type(torch.bool)
        # key_padding_mask[tokens == self.padding_idx] = True
        return key_padding_mask
