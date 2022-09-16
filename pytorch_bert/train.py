import tokenize
import torch
from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F

PRETRAINED_PATH = "./pretrained"

#config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-cased')  # Download configuration from S3 and cache.
#config.output_attentions = True
#config.output_hidden_states = True
#model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased', output_attentions=True, config=config)  # Update configuration during loading

# model = BertModel.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('distilbert-base-cased')


class CustomBERTModel(BertPreTrainedModel):
   def __init__(self, config, num_class):
      super(CustomBERTModel, self).__init__(config)
      self.bert = BertModel(config)
      self.linear = nn.Linear(config.hidden_size, num_class)

#model = CustomBERTModel.from_pretrained('distilbert-base-cased', num_class=2)

#model = AutoModelWithLMHead.from_pretrained("marrrcin/PolBERTa-base-polish-cased-v1")
model = AutoModelForCausalLM.from_pretrained("marrrcin/PolBERTa-base-polish-cased-v1")

print(f"{model}")

base_model = model.roberta


text = ["I am flash", "123"]
# text = "I am flash"

def build_tokenizer():
   from transformers import RobertaTokenizerFast
   tokenizer = RobertaTokenizerFast.from_pretrained(PRETRAINED_PATH)
   return tokenizer

tokenizer = build_tokenizer()

'''
indexed_tokens = tokenizer.encode_plus(text)
res = tokenizer(text, max_length=100, padding='max_length', truncation=True)
segments_ids = res["input_ids"]

print(f"{indexed_tokens=}")

# Convert inputs to PyTorch tensors
segments_tensors = torch.tensor([segments_ids])
print(f"{segments_tensors=}")
'''

enc = tokenizer.encode_plus(text)
out = model(torch.tensor(enc["input_ids"]).unsqueeze(0), torch.tensor(enc["attention_mask"]).unsqueeze(0))
print(f"{out=}")

token_representations = base_model(torch.tensor(enc["input_ids"]).unsqueeze(0))[0][0]
print(f"{token_representations=}")


@torch.jit.script
def mish(input):
    return input * torch.tanh(F.softplus(input))
  
class Mish(nn.Module):
    def forward(self, input):
        return mish(input)

class PolBERTaSentimentModel(nn.Module):
    def __init__(self, base_model, n_classes, base_model_output_size=768, dropout=0.05):
        super().__init__()
        self.base_model = base_model
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_model_output_size, base_model_output_size),
            Mish(),
            nn.Dropout(dropout),
            nn.Linear(base_model_output_size, n_classes)
        )
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, input_, *args):
        X, attention_mask = input_
        hidden_states, _ = self.base_model(X, attention_mask=attention_mask)
        
        # here I use only representation of <s> token, but you can easily use more tokens,
        # maybe do some pooling / RNNs... go crazy here!
        return self.classifier(hidden_states[:, 0, :])


classifier = PolBERTaSentimentModel(AutoModelWithLMHead.from_pretrained("marrrcin/PolBERTa-base-polish-cased-v1").roberta, 2)









from torch.utils.data.dataset import Dataset
import numpy as np
max_seq_length = 256
class text_dataset(Dataset):
   def __init__(self, x_y_list):
      self.x_y_list = x_y_list
        
   def __getitem__(self, index):
      tokenized_review = tokenizer.tokenize(self.x_y_list[0][index])
     
      if len(tokenized_review) > max_seq_length:
         tokenized_review = tokenized_review[:max_seq_length]
         
      ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review)
      padding = [0] * (max_seq_length - len(ids_review))
      ids_review += padding
     
      assert len(ids_review) == max_seq_length
      #print(ids_review)
      ids_review = torch.tensor(ids_review)
      sentiment = self.x_y_list[1][index]
      list_of_labels = [torch.from_numpy(np.array(sentiment))]
      return ids_review, list_of_labels[0]
    
   def __len__(self):
      return len(self.x_y_list[0])


import torch.optim as optim
from torch.optim import lr_scheduler

# lrlast = .001
# lrmain = .00001
# optim1 = optim.Adam(
#     [
#         {"params":model.bert.parameters(),"lr": lrmain},
#         {"params":model.classifier.parameters(), "lr": lrlast},       
#    ])
# optimizer_ft = optim1

