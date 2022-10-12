import torch
import torch.nn as nn
import torch.nn.functional as F
import string


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tensor = torch.rand(3,4)
tensor = tensor.to('cuda')
print(f"Device tensor is stored on: {tensor.device}")


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        :param input_size: number of input coming in
        :param hidden_size: number of he hidden units
        :param output_size: size of the output
        '''
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        #LSTM
        self.lstm = nn.LSTM(input_size, hidden_size).to(device)
        self.hidden2Cat = nn.Linear(hidden_size, output_size).to(device)
        self.hidden = self.init_hidden()

    def forward(self, input):

        lstm_out, self.hidden = self.lstm(input, self.hidden)
        output = self.hidden2Cat(lstm_out[-1]) #many to one
        output = F.log_softmax(output, dim=1)

        return output

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size).to(device),
                torch.zeros(1, 1, self.hidden_size).to(device))

def export_model_to_onnx(model):
   # dummy_input = torch.randn(1, 3, 224, 224)
   # Let’s also define the input and output names.
   #input_names = [ "actual_input" ]
   #output_names = [ "output" ]
   dummy_input = torch.randn(1, 1, 57).to(device)
   input_names = [ "actual_input" ]
   output_names = [ "output" ]
   torch.onnx.export(model, 
                  dummy_input,
                  "names.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  )


def demo():
   all_letters = string.ascii_letters + " .,;'"
   n_letters = len(all_letters)
   all_categories = ['English', 'Rusin']
   n_categories = len(all_categories)
   n_hidden = 128
   model = RNN(n_letters, n_hidden, n_categories)
   model.eval()
   print(model)
   """"""
   export_model_to_onnx(model)


   
if __name__ == '__main__':
   demo()

