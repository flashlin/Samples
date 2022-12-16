import torch

from ml.gnmt.beam_search import SequenceGenerator


def batch_padded_sequences(seq, padding_idx, batch_first=False, sort=False):
    if sort:
        key = lambda item: len(item[1])
        indices, seq = zip(*sorted(enumerate(seq), key=key, reverse=True))
    else:
        indices = range(len(seq))

    lengths = [len(sentence) for sentence in seq]
    batch_length = max(lengths)
    seq_tensor = torch.LongTensor(batch_length, len(seq)).fill_(padding_idx)
    for idx, sentence in enumerate(seq):
        end_seq = lengths[idx]
        seq_tensor[:end_seq, idx].copy_(sentence[:end_seq])
    if batch_first:
        seq_tensor = seq_tensor.t()
    return seq_tensor, lengths, indices


class Translator:
    def __init__(self, model, tok,
                 beam_size=5,
                 len_norm_factor=0.6,
                 len_norm_const=5.0,
                 cov_penalty_factor=0.1,
                 max_seq_len=50,
                 cuda=False):

        self.model = model
        self.tok = tok
        self.insert_target_start = [tok.bos_idx]
        self.insert_src_start = [tok.bos_idx]
        self.insert_src_end = [tok.eos_idx]
        self.batch_first = model.batch_first
        self.cuda = cuda
        self.beam_size = beam_size

        self.generator = SequenceGenerator(
            model=self.model,
            bos_idx=tok.bos_idx,
            eos_idx=tok.eos_idx,
            beam_size=beam_size,
            max_seq_len=max_seq_len,
            cuda=cuda,
            len_norm_factor=len_norm_factor,
            len_norm_const=len_norm_const,
            cov_penalty_factor=cov_penalty_factor)

    def translate(self, input_sentences):
        stats = {}
        batch_size = len(input_sentences)
        beam_size = self.beam_size

        src_tok = [torch.tensor(self.tok.encode_to_tokens(line)) for line in input_sentences]

        bos = [self.insert_target_start] * (batch_size * beam_size)
        bos = torch.LongTensor(bos)
        if self.batch_first:
            bos = bos.view(-1, 1)
        else:
            bos = bos.view(1, -1)

        src = batch_padded_sequences(src_tok, self.tok.padding_idx, self.batch_first, sort=True)
        src, src_length, indices = src

        src_length = torch.LongTensor(src_length)
        stats['total_enc_len'] = int(src_length.sum())

        if self.cuda:
            src = src.cuda()
            src_length = src_length.cuda()
            bos = bos.cuda()

        with torch.no_grad():
            context = self.model.encode_to_tokens(src, src_length)
            context = [context, src_length, None]

            if beam_size == 1:
                generator = self.generator.greedy_search
            else:
                generator = self.generator.beam_search

            preds, lengths, counter = generator(batch_size, bos, context)

        preds = preds.cpu()
        lengths = lengths.cpu()

        output = []
        for idx, pred in enumerate(preds):
            end = lengths[idx] - 1
            pred = pred[1: end]
            pred = pred.tolist()
            out = self.tok.decode(pred)
            output.append(out)

        stats['total_dec_len'] = int(lengths.sum())
        stats['iters'] = counter

        output = [output[indices.index(i)] for i in range(len(output))]
        return output, stats
