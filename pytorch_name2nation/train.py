# from __future__ import unicode_literals, print_function, division
from io import open
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# clearing output
from IPython.display import clear_output

from service_api.app.name2lang import findFiles, readLines, NameToNationClassify, dataloader

languages = []
X = []
y = []

for filename in findFiles('data/names/*.txt'):
    lang = os.path.splitext(os.path.basename(filename))[0]
    if not lang in languages:
        languages.append(lang)
    languages.append(lang)
    lines = readLines(filename)
    for name in lines:
        X.append(name)
        y.append(lang)

with open("name2lang.txt", 'r') as f:
    # read the dataset
    for line in f:
        line = line.split(",")
        name = line[0].strip()
        lang = line[1].strip()
        if not lang in languages:
            languages.append(lang)
        if name in X:
            continue
        X.append(name)
        y.append(lang)

n_languages = len(languages)
print(f"{n_languages=}")

"""## Train Test Split"""
# split the data 70 30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
print("Training Data: ", len(X_train))
print("Testing Data: ", len(X_test))

"""# Encoding Names and Languages"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


def train_setup(net, lr=0.01, n_batches=100, batch_size=10, momentum=0.9, display_freq=20):
    model = net.model
    criterion = nn.NLLLoss()  # define a loss function
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)  # define a optimizer
    loss_arr = np.zeros(n_batches + 1)
    # iterate through all the batches
    for i in range(n_batches):
        loss_arr[i + 1] = (loss_arr[i] * i + net.train(opt, criterion, batch_size, X_train, y_train)) / (i + 1)

        if i % display_freq == display_freq - 1:
            #clear_output(wait=True)
            print("Iteration number ", i + 1, "Top - 1 Accuracy:",
                  round(net.eval(len(X_test), 1, X_test, y_test), 4),
                  'Top-2 Accuracy:',
                  round(net.eval(len(X_test), 2, X_test, y_test), 4), 'Loss:',
                  round(loss_arr[i], 4)
                  )
            # plt.figure()
            # plt.plot(loss_arr[1:i], "-*")
            # plt.xlabel("Iteration")
            # plt.ylabel("Loss")
            # plt.show()
            #print("\n\n")


# declare all the parameters
model_dir = "service_api/models"
net = NameToNationClassify(model_dir)

model_file = "lstm1.model"
net.try_load_state(model_file)

train_setup(net, lr=0.0005, n_batches=100, batch_size=256)
net.save_state(model_file)
