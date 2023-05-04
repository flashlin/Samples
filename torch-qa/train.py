import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

class DistilBertForQA(nn.Module):
    def __init__(self):
        super(DistilBertForQA, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.qa_outputs = nn.Linear(self.distilbert.config.hidden_size, 2)
        self.dropout = nn.Dropout(self.distilbert.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dropout(pooled_output)
        qa_logits = self.qa_outputs(pooled_output)
        return qa_logits



questions = [
    "What is the capital of Taiwan?",
    "Who invented the telephone?",
    "When was the first iPhone released?"
]

contexts = [
    "Taipei is the capital of Taiwan.",
    "Alexander Graham Bell invented the telephone.",
    "The first iPhone was released in 2007."
]

answers = [
    ("Taipei", "Taiwan"),
    ("Alexander Graham Bell", "the telephone"),
    ("2007", "the first iPhone")
]

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def prepare_data(question, context, answer):
    encoding = tokenizer.encode_plus(question, context, return_tensors='pt', max_length=512, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    start_positions = torch.tensor([tokenizer.encode(answer[0], add_specialtokens=False).offsets[0][0]], dtype=torch.long)
    end_positions = torch.tensor([tokenizer.encode(answer[1], add_special_tokens=False).offsets[-1][-1] - 1], dtype=torch.long)
    return input_ids, attention_mask, start_positions, end_positions

train_data = []
for q, c, a in zip(questions, contexts, answers):
    input_ids, attention_mask, start_positions, end_positions = prepare_data(q, c, a)
    train_data.append((input_ids, attention_mask, start_positions, end_positions))

model = DistilBertForQA()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()


for epoch in range(3):
    for input_ids, attention_mask, start_positions, end_positions in train_data:
        optimizer.zero_grad()
        qa_logits = model(input_ids, attention_mask)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        start_loss = loss_fn(start_logits, start_positions)
        end_loss = loss_fn(end_logits, end_positions)
        loss = start_loss + end_loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: loss = {loss.item():.3f}")

def qa(model, question, context):
    encoding = tokenizer.encode_plus(question, context, return_tensors='pt', max_length=512, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    qa_logits = model(input_ids, attention_mask)
    start_logits, end_logits = qa_logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).softmax(dim=-1).tolist()[0]
    end_logits = end_logits.squeeze(-1).softmax(dim=-1).tolist()[0]
    start_index = start_logits.index(max(start_logits))
    end_index = end_logits.index(max(end_logits))
    answer_tokens = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index+1]))
    return answer_tokens

qa(model, "where is the capital of taiwan")    

