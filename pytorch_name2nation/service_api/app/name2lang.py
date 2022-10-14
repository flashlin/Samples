import os, string, random, time, math
import torch 
import torch.nn as nn
import torch.optim as optim
import glob
import unicodedata
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
   #print(f"\r\n\r\n{predict(net, 'Elon Musk')=}")


class NameToNationClassify:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.languages = languages = read_languages(f"{model_dir}/langs.txt")
        n_languages = len(languages)
        n_hidden = 128
        self.model = Lstm(n_letters, n_hidden, n_languages)

    def load_state(self):
        model = self.model
        model.load_state_dict(torch.load(f"{self.model_dir}/lstm.model"))
        model.eval()

    def name_rep(self, name):
        rep = torch.zeros(len(name), 1, n_letters)
        for index, letter in enumerate(name):
            pos = all_letters.find(letter)
            rep[index][0][pos] = 1
        return rep

    def infer(self, name):
        model = self.model
        model.eval()
        name_ohe = self.name_rep(name)
        hidden = model.init_hidden()
        for i in range(name_ohe.size()[0]):
            output, hidden = model(name_ohe[i], hidden)
        return output

    def predict(self, name):
       output = self.infer(name)
       val, indices = output.topk(3) #get top 3
       lgs = [n for n in indices.detach().numpy()]
       lgs = [self.languages[x] for x in lgs[0]]
       return lgs

