from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

from pdf_utils import splitting_documents_into_texts, load_txt_documents_from_directory

tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

documents = load_txt_documents_from_directory('./news')
texts = splitting_documents_into_texts(documents)

# import chromadb
# chroma_client = chromadb.Client()

def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

context = "MingMing's birth is 2023-03-01. SmallPig have brother, his name is Flash. Flash and wife life "
question = "what name is the SmallPig's sister?"
answer = answer_question(question, context)
print(f'{question} {answer=}')

