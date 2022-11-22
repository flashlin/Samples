from nltk.translate import bleu_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import utils
import torch
import os
from nmt_model import NMTModel
from nmt_dataset import NMTDataset
import nmt_dataset

chencherry = bleu_score.SmoothingFunction()
args = utils.get_args()


class NMTSampler:
    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model

    def apply_to_batch(self, batch_dict):
        self._last_batch = batch_dict
        y_pred = self.model(x_source=batch_dict['x_source'],
                            x_source_lengths=batch_dict['x_source_length'],
                            target_sequence=batch_dict['x_target'])
        self._last_batch['y_pred'] = y_pred

        attention_batched = np.stack(self.model.decoder._cached_p_attn).transpose(1, 0, 2)
        self._last_batch['attention'] = attention_batched

    def _get_source_sentence(self, index, return_string=True):
        indices = self._last_batch['x_source'][index].cpu().detach().numpy()
        vocab = self.vectorizer.source_vocab
        return utils.sentence_from_indices(indices, vocab, return_string=return_string)

    def _get_reference_sentence(self, index, return_string=True):
        indices = self._last_batch['y_target'][index].cpu().detach().numpy()
        vocab = self.vectorizer.target_vocab
        return utils.sentence_from_indices(indices, vocab, return_string=return_string)

    def _get_sampled_sentence(self, index, return_string=True):
        _, all_indices = torch.max(self._last_batch['y_pred'], dim=2)
        sentence_indices = all_indices[index].cpu().detach().numpy()
        vocab = self.vectorizer.target_vocab
        return utils.sentence_from_indices(sentence_indices, vocab, return_string=return_string)

    def get_ith_item(self, index, return_string=True):
        output = {"source": self._get_source_sentence(index, return_string=return_string),
                  "reference": self._get_reference_sentence(index, return_string=return_string),
                  "sampled": self._get_sampled_sentence(index, return_string=return_string),
                  "attention": self._last_batch['attention'][index]}

        reference = output['reference']
        hypothesis = output['sampled']

        if not return_string:
            reference = " ".join(reference)
            hypothesis = " ".join(hypothesis)

        output['bleu-4'] = bleu_score.sentence_bleu(references=[reference],
                                                    hypothesis=hypothesis,
                                                    smoothing_function=chencherry.method1)

        return output


dataset = NMTDataset.load_dataset_and_load_vectorizer(args.dataset_csv,
                                                      args.vectorizer_file)
vectorizer = dataset.get_vectorizer()
# create model
model = NMTModel(source_vocab_size=len(vectorizer.source_vocab),
                 source_embedding_size=args.source_embedding_size,
                 target_vocab_size=len(vectorizer.target_vocab),
                 target_embedding_size=args.target_embedding_size,
                 encoding_size=args.encoding_size,
                 target_bos_index=vectorizer.target_vocab.begin_seq_index)

# load from checkpoint or create new one
# if args.reload_from_files and os.path.exists(args.model_state_file):
model.load_state_dict(torch.load(args.model_state_file))
#     print("Reloaded model")
# else:
#     print("New model")

model = model.eval().to(args.device)

sampler = NMTSampler(vectorizer, model)

dataset.set_split('test')
batch_generator = nmt_dataset.generate_nmt_batches(dataset,
                                                   batch_size=args.batch_size,
                                                   device=args.device)

test_results = []
for batch_dict in batch_generator:
    sampler.apply_to_batch(batch_dict)
    for i in range(args.batch_size):
        test_results.append(sampler.get_ith_item(i, False))
plt.hist([r['bleu-4'] for r in test_results], bins=100)
print(np.mean([r['bleu-4'] for r in test_results]), np.median([r['bleu-4'] for r in test_results]))

plt.show()

all_results = []
for i in range(args.batch_size):
    all_results.append(sampler.get_ith_item(i, False))
top_results = [x for x in all_results if x['bleu-4'] > 0.5]
for sample in top_results:
    plt.figure()
    target_len = len(sample['sampled'])
    source_len = len(sample['source'])

    attention_matrix = sample['attention'][:target_len, :source_len + 2].transpose()  # [::-1]
    ax = sns.heatmap(attention_matrix, center=0.0)
    ylabs = ["<BOS>"] + sample['source'] + ["<EOS>"]
    # ylabs = sample['source']
    # ylabs = ylabs[::-1]
    ax.set_yticklabels(ylabs, rotation=0)
    ax.set_xticklabels(sample['sampled'], rotation=90)
    ax.set_xlabel("Target Sentence")
    ax.set_ylabel("Source Sentence\n\n")
plt.show()


def get_source_sentence(vectorizer, batch_dict, index):
    indices = batch_dict['x_source'][index].cpu().data.numpy()
    vocab = vectorizer.source_vocab
    return sentence_from_indices(indices, vocab)


def get_true_sentence(vectorizer, batch_dict, index):
    return sentence_from_indices(batch_dict['y_target'].cpu().data.numpy()[index], vectorizer.target_vocab)


def get_sampled_sentence(vectorizer, batch_dict, index):
    y_pred = model(x_source=batch_dict['x_source'],
                   x_source_lengths=batch_dict['x_source_length'],
                   target_sequence=batch_dict['x_target'],
                   sample_probability=1.0)
    return sentence_from_indices(torch.max(y_pred, dim=2)[1].cpu().data.numpy()[index], vectorizer.target_vocab)


def get_all_sentences(vectorizer, batch_dict, index):
    return {"source": get_source_sentence(vectorizer, batch_dict, index),
            "truth": get_true_sentence(vectorizer, batch_dict, index),
            "sampled": get_sampled_sentence(vectorizer, batch_dict, index)}


def sentence_from_indices(indices, vocab, strict=True):
    ignore_indices = set([vocab.mask_index, vocab.begin_seq_index, vocab.end_seq_index])
    out = []
    for index in indices:
        if index == vocab.begin_seq_index and strict:
            continue
        elif index == vocab.end_seq_index and strict:
            return " ".join(out)
        else:
            out.append(vocab.lookup_index(index))
    return " ".join(out)


results = get_all_sentences(vectorizer, batch_dict, 1)
print(results)
