from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

# pip install --upgrade protobuf

# Define the training data
train_data = [
    {
        "context": "The capital of France is Paris.",
        "question": "What is the capital of France?",
        "answer": "Paris",
    },
    {
        "context": "The Great Pyramid of Giza is in Egypt.",
        "question": "Where is the Great Pyramid of Giza?",
        "answer": "Egypt",
    },
    {
        "context": "The tallest mountain in the world is Mount Everest.",
        "question": "What is the tallest mountain in the world?",
        "answer": "Mount Everest",
    }
]

# Load the DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

# Set the model to training mode
model.train()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Train the model
for epoch in range(3):
    for data in train_data:
        # Encode the training data
        encoding = tokenizer.encode_plus(data['question'], data['context'], return_tensors='pt')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # Locate the start and end positions of the answer
        start_positions = input_ids[0].tolist().index(tokenizer.encode(data['answer'], add_special_tokens=False)[0])
        end_positions = start_positions + len(tokenizer.encode(data['answer'], add_special_tokens=False)) - 1
        
        # Compute the loss
        # loss, _, _ = model(input_ids, attention_mask=attention_mask, start_positions=torch.tensor([start_positions]), end_positions=torch.tensor([end_positions]))
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=torch.tensor([start_positions]), end_positions=torch.tensor([end_positions]))
        loss = outputs.loss
        
        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Define a function to perform QA using the trained model
def answer_question(question, context):
    # Set the model to evaluation mode
    model.eval()
    
    # Encode the question and context
    encoding = tokenizer.encode_plus(question, context, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # Compute the answer using the trained model
    with torch.no_grad():
        #start_scores, end_scores = model(input_ids, attention_mask=attention_mask)
        outputs = model(input_ids, attention_mask=attention_mask)

        #answer_start = torch.argmax(outputs.start_logits)
        #answer_end = torch.argmax(outputs.end_logits)
        #answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end+1])
        start_index = torch.argmax(outputs.start_logits)
        end_index = torch.argmax(outputs.end_logits)
        
    # Locate the start and end positions of the answer
    # start_index = torch.argmax(start_scores)
    # end_index = torch.argmax(end_scores)
    
    # Decode the answer from the input IDs
    answer_ids = input_ids[0][start_index:end_index+1]
    answer = tokenizer.decode(answer_ids)
    
    print(f'{question} ==> "{answer}"')
    return answer

# Example usage of the answer_question function
answer_question("What is the capital of France?", "The capital of France is Paris.")
answer_question("What is the capital of Taiwan?", "The capital of France is Paris.")