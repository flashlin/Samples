from io import open
import os, string, random, time, math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
import torch.optim as optim

#clearing output
from IPython.display import clear_output

languages = []
data = []
X = []
y = []

with open("name2lang.txt", 'r') as f:
    #read the dataset
    for line in f:
        line = line.split(",")
        name = line[0].strip()
        lang = line[1].strip()
        if not lang in languages:
            languages.append(lang)
        X.append(name)
        y.append(lang)
        data.append((name, lang))

n_languages = len(languages)
print(f"{n_languages=}")


def convert_to_language(idx):
   return languages[idx]


"""## Train Test Split"""
#split the data 70 30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123, stratify = y)
print("Training Data: ", len(X_train))
print("Testing Data: ", len(X_test))




"""# Encoding Names and Languages"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


#get all the letters
all_letters = string.ascii_letters + ".,;"
n_letters = len(all_letters)
print("Number of letters: ", n_letters)

def name_rep(name):
    rep = torch.zeros(len(name), 1, n_letters)
    for index, letter in enumerate(name):
        pos = all_letters.find(letter)
        rep[index][0][pos] = 1
    return rep

#function to create lang representation
def nat_rep(lang):
    return torch.tensor([languages.index(lang)], dtype=torch.long, device=device)

print(f'{name_rep("Kumar")=}')
print(f'{nat_rep("Irish")=}')




"""# Basic EDA Analysis"""

#check the distribution of the languages

count = {}
for l in languages:
    count[l] = 0

for d in data:
    count[d[1]] += 1

#plot the distribution
plt.style.use("seaborn")
plt_ = sns.barplot(x=list(count.keys()), y=list(count.values()))
plt_.set_xticklabels(plt_.get_xticklabels(), rotation = 90)
plt.show()



"""# RNN Network"""

#define a basic rnn network

class RNN_net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_net, self).__init__()
        #declare the hidden size for the network
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) #input to hidden layer
        self.i2o = nn.Linear(input_size + hidden_size, output_size) #input to output layer
        self.softmax = nn.LogSoftmax(dim = 1) #softmax for classification
    
    def forward(self, input_, hidden):
        combined = torch.cat((input_, hidden), 1) #concatenate tensors on column wise
        hidden = self.i2h(combined) #generate hidden representation
        output = self.i2o(combined) #generate output representation
        output = self.softmax(output) #get the softmax label
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

#declare the size of the hidden layer representation
n_hidden = 128

#create a object of the class
net = RNN_net(n_letters, n_hidden, n_languages)

#function to make inference

def infer(net, name):
    net.eval()
    name_ohe = name_rep(name)
    hidden = net.init_hidden()
    for i in range(name_ohe.size()[0]):
        output, hidden = net(name_ohe[i], hidden)
    return output


#before training the network, make a inference to test the network

output = infer(net, "Adam")
index = torch.argmax(output)
print(output, index)

"""# Evaluate Basic RNN Model
- Create a evaluation setup
"""

#create a dataloader

def dataloader(npoints, X_, y_):
    to_ret = []
    for i in range(npoints):
        index_ = np.random.randint(len(X_))
        name, lang = X_[index_], y_[index_] #get the data at the random index
        to_ret.append((name, lang, name_rep(name), nat_rep(lang)))

    return to_ret

dataloader(1, X_train, y_train)


#create a function to evaluate model

def eval(net, n_points, k, X_, y_):
     data_ = dataloader(n_points, X_, y_)
     correct = 0

     for name, language, name_ohe, lang_rep in data_:
         output = infer(net, name) #prediction
         val, indices = output.topk(k) #get the top k predictions

         if lang_rep in indices:
             correct += 1
    
     accuracy = correct/n_points
     return accuracy

#basic model evaluation - top 3 accuracy
eval(net, 100, 3, X_test, y_test)


"""# Training SetUp"""

#function to train the data

def train(net, opt, criterion, n_points):
    opt.zero_grad()
    total_loss = 0
    
    data_ = dataloader(n_points, X_train, y_train)
    
    for name, language, name_ohe, lang_rep in data_:
        hidden = net.init_hidden()

        for i in range(name_ohe.size()[0]):
            output, hidden = net(name_ohe[i], hidden)
            
        loss = criterion(output, lang_rep)
        loss.backward(retain_graph=True)
        
        total_loss += loss
        
    opt.step()       
            
    return total_loss/n_points

def train_setup(net, lr = 0.01, n_batches = 100, batch_size = 10, momentum = 0.9, display_freq = 20):
    criterion = nn.NLLLoss() #define a loss function
    opt = optim.SGD(net.parameters(), lr = lr, momentum = momentum) #define a optimizer
    loss_arr = np.zeros(n_batches + 1)
    #iterate through all the batches
    for i in range(n_batches):
        loss_arr[i + 1] = (loss_arr[i]*i + train(net, opt, criterion, batch_size))/(i + 1)

        if i%display_freq == display_freq - 1:
            clear_output(wait = True)

            print("Iteration number ", i + 1, "Top - 1 Accuracy:", round(eval(net, len(X_test), 1, X_test, y_test),4), 
                  'Top-2 Accuracy:', round(eval(net, len(X_test), 2, X_test, y_test),4), 'Loss:', round(loss_arr[i],4))
            plt.figure()
            plt.plot(loss_arr[1:i], "-*")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.show()
            print("\n\n")

#declare all the parameters

n_hidden = 128
net = RNN_net(n_letters, n_hidden, n_languages)
#train_setup(net, lr = 0.0005, n_batches = 100, batch_size = 256)



"""# LSTM Implementation"""
#LSTM class
class LSTM_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_net, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTM(input_size, hidden_size) #LSTM cell
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 2)

    def forward(self, input_, hidden):
        out, hidden = self.lstm_cell(input_.view(1, 1, -1), hidden)
        output = self.h2o(hidden[0])
        output = self.softmax(output)
        return output.view(1, -1), hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

#create hyperparameters
n_hidden = 128
net = LSTM_net(n_letters, n_hidden, n_languages)


def predict(net, name):
   output = infer(net, name)
   val, indices = output.topk(3) #get top 3
   lgs = [n for n in indices.detach().numpy()]
   lgs = [convert_to_language(x) for x in lgs[0]]
   return lgs


model_path = "./models/lstm.model"
if os.path.exists(model_path):
   net.load_state_dict(torch.load(model_path))
   print(f"{net.eval()=}")
   rc = predict('April')
   print(f"{rc=}")
else:
   train_setup(net, lr = 0.0005, n_batches = 100, batch_size = 256)
   torch.save(net.state_dict(), model_path)



"""# GRU Unit"""
class GRU_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU_net, self).__init__()
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRU(input_size, hidden_size) #gru cell
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 2)
    
    def forward(self, input_, hidden):
        out, hidden = self.gru_cell(input_.view(1, 1, -1), hidden)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output.view(1, -1), hidden
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

#hyperparameters
n_hidden = 128
net = LSTM_net(n_letters, n_hidden, n_languages)
#train_setup(net, lr = 0.0005, n_batches = 100, batch_size = 256)

