import numpy as np
from transformers import DistilBertModel, DistilBertTokenizer, GPT2Tokenizer
import faiss
import torch

# Load DistilBERT model and tokenizer
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load GPT-2 tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Define a function to embed text using DistilBERT
def embed_text(text):
    input_ids = distilbert_tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        embeddings = distilbert_model(input_ids).last_hidden_state[:,0,:].numpy()
    return embeddings

# Define a list of documents to use for QA
documents = ["Document 1 text", "Document 2 text", "Document 3 text"]

# Embed the documents using DistilBERT
embeddings = [embed_text(doc) for doc in documents]
embeddings = np.concatenate(embeddings)

# Create a FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Define a function to perform QA using the FAISS index and GPT-2 tokenizer
def answer_question(question):
    # Encode the question using the GPT-2 tokenizer
    input_ids = gpt2_tokenizer.encode(question, return_tensors='pt')
    
    # Embed the question using DistilBERT
    question_embedding = embed_text(question)
    
    # Search for the most similar document using the FAISS index
    D, I = index.search(question_embedding, 1)
    most_similar_document = documents[I[0][0]]
    
    # Return the most similar document as the answer
    return most_similar_document

# Example usage of the answer_question function
answer = answer_question("What is the text of document 2?")
print(answer)


from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2')
















from transformers import DistilBertTokenizer, DistilBertModel
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_embeddings(text):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_states = outputs[0].detach().numpy()
    return last_hidden_states



from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def get_input_ids(question):
    inputs = tokenizer.encode_plus(question, return_tensors='pt')
    return inputs['input_ids']

def answer_question(question, embeddings, index, documents):
    input_ids = get_input_ids(question)
    query_embedding = get_embeddings(question)
    doc_ids = index.search(query_embedding, k=1)[1].tolist()
    top_doc_id = doc_ids[0][0]
    document = documents[top_doc_id]
    input_ids += tokenizer.encode(document, add_special_tokens=True)
    inputs = tokenizer.encode_plus(document, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=1000, do_sample=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
