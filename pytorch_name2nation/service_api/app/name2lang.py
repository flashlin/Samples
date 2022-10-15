import os, string, random, time, math

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import unicodedata
from matplotlib import pyplot as plt

from lstm import Lstm

all_letters = string.ascii_letters + ".,;"
n_letters = len(all_letters)


def read_languages(lang_file_path):
    languages = []
    with open(lang_file_path, 'r') as f:
        for line in f:
            lang = line.strip()
            if not lang in languages:
                languages.append(lang)
            languages.append(lang)
    return languages


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def findFiles(path): return glob.glob(path)


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def get_train_languages(train_data_dir):
    languages = []
    for filename in findFiles(f'{train_data_dir}/names/*.txt'):
        lang = os.path.splitext(os.path.basename(filename))[0]
        if not lang in languages:
            languages.append(lang)
    return languages


def load_model_state(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # print(f"\r\n\r\n{predict(net, 'Elon Musk')=}")


class NameToNationClassify:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.languages = languages = read_languages(f"{model_dir}/langs.txt")
        n_languages = len(languages)
        n_hidden = 128
        self.model = Lstm(n_letters, n_hidden, n_languages)

    def load_state(self, model_file):
        model = self.model
        model.load_state_dict(torch.load(f"{self.model_dir}/{model_file}"))
        # model.eval()

    def try_load_state(self, model_file):
        if not os.path.exists(f"{self.model_dir}/{model_file}"):
            return False
        self.load_state(model_file)
        return True

    def save_state(self, model_file):
        torch.save(self.model.state_dict(), f"{self.model_dir}/{model_file}")

    def name_rep(self, name):
        rep = torch.zeros(len(name), 1, n_letters)
        for index, letter in enumerate(name):
            pos = all_letters.find(letter)
            rep[index][0][pos] = 1
        return rep

    def nat_rep(self, lang):
        return torch.tensor([self.languages.index(lang)], dtype=torch.long)

    def infer(self, name):
        model = self.model
        model.eval()
        name_ohe = self.name_rep(name)
        hidden = model.init_hidden()
        for i in range(name_ohe.size()[0]):
            output, hidden = model(name_ohe[i], hidden)
        return output

    def eval(self, batch_size, k, X_, y_):
        data_ = dataloader(batch_size, X_, y_, self.name_rep, self.nat_rep)
        correct = 0
        for name, language, name_ohe, lang_rep in data_:
            output = self.infer(name)  # prediction
            val, indices = output.topk(k)  # get the top k predictions
            if lang_rep in indices:
                correct += 1
        accuracy = correct / batch_size
        return accuracy

    def train(self, opt, criterion, batch_size, X_train, y_train):
        opt.zero_grad()
        total_loss = 0
        data_ = dataloader(batch_size, X_train, y_train, self.name_rep, self.nat_rep)
        for name, language, name_ohe, lang_rep in data_:
            hidden = self.model.init_hidden()
            for i in range(name_ohe.size()[0]):
                output, hidden = self.model(name_ohe[i], hidden)
            loss = criterion(output, lang_rep)
            loss.backward(retain_graph=True)
            total_loss += loss
        opt.step()
        return total_loss / batch_size

    def predict(self, name):
        self.model.eval()
        output = self.infer(name)
        val, indices = output.topk(3)  # get top 3
        lgs = [n for n in indices.detach().numpy()]
        lgs = [self.languages[x] for x in lgs[0]]
        return lgs


"""
    data: [(name, lang)]
"""
def draw_distribution_of_the_languages(languages, data):
    count = {}
    for l in languages:
        count[l] = 0

    for d in data:
        count[d[1]] += 1

    # plot the distribution
    plt.style.use("seaborn")
    plt_ = sns.barplot(x=list(count.keys()), y=list(count.values()))
    plt_.set_xticklabels(plt_.get_xticklabels(), rotation=90)
    plt.show()


def dataloader(batch_size, X_, y_, name_rep, nat_rep):
    to_ret = []
    for i in range(batch_size):
        index_ = np.random.randint(len(X_))
        name, lang = X_[index_], y_[index_]  # get the data at the random index
        to_ret.append((name, lang, name_rep(name), nat_rep(lang)))
    return to_ret
