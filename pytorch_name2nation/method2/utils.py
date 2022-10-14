import glob
import math
import os
import random
import time

import unicodedata
import string
import torch
import torch.nn as nn
import torch.optim as optim

from lstm import RNN


def findFiles(path): return glob.glob(path)


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def readNames():
    category_lines = {}
    all_categories = []
    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    return category_lines, all_categories


def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def info(msg):
    print(f"{bcolors.OKGREEN}{msg}{bcolors.ENDC}")

category_lines, all_categories = readNames()
n_categories = len(all_categories)
info(f"{all_categories=}")

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(arr):
    return arr[random.randint(0, len(arr) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


criterion = nn.NLLLoss()
learning_rate = 0.001  # If you set this too high, it might explode. If too low, it might not learn
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)  # LSTM model
optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)


def train(category_tensor, line_tensor):
    rnn.zero_grad()
    rnn.hidden = rnn.init_hidden()

    output = rnn(line_tensor)[-1]

    loss = criterion(output.unsqueeze(0), category_tensor)
    loss.backward()

    optimizer.step()
    return output.unsqueeze(0), loss.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def export_model_to_onnx(model):
   info(f"{model=}")
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   dummy_input = torch.randn(1, 1, 57).to(device)
   input_names = [ "actual_input" ]
   output_names = [ "output" ]
   torch.onnx.export(model, 
                  dummy_input,
                  "names1.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  )

def save_model():
    rnn.eval()
    export_model_to_onnx(rnn)
