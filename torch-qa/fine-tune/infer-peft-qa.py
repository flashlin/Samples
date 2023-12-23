import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from finetune_utils import load_finetune_config
from langchain_lit import load_markdown_documents, LlmEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

config = load_finetune_config()
device = "cuda"
EMB_MODEL = "bge-base-en"


def load_vector_store():
    print("loading data")
    docs = load_markdown_documents("./data")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=35)
    all_splits = text_splitter.split_documents(docs)
    llm_embedding = LlmEmbedding(f"../models/{EMB_MODEL}")
    print("loading vector")
    vectorstore = Chroma.from_documents(documents=all_splits,
                                        embedding=llm_embedding.embedding)
    return vectorstore


def clean_prompt_resp(resp: str):
    after_inst = resp.split("[/INST]", 1)[-1]
    s2 = after_inst.split("[INST]", 1)[0]
    return s2.split('[/INST]', 1)[0]


model_name = config['model_name']
base_model = f"../models/{model_name}"
peft_model_id = f"./outputs/{model_name}-qlora"
if not os.path.exists(f"{peft_model_id}/adapter_config.json"):
    peft_model_id = None

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    # return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    # trust_remote_code=True,
    local_files_only=True,
)
model.load_adapter(peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# model = PeftModel.from_pretrained(model, PEFT_MODEL)

generation_config = model.generation_config
generation_config.max_new_tokens = 4096
generation_config.temperature = 0.01  # 0.7
generation_config.top_p = 2
generation_config.do_sample = True
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

SYSTEM_PROMPT = """"""
prompt_template = """<s>[INST] {user_input} [/INST]"""

task = "text-generation"
pipe = pipeline(
    task=task,
    model=model,
    tokenizer=tokenizer,
    # max_length=4096,
    temperature=0.01,
    top_p=2,
    repetition_penalty=1.15,
    return_full_text=True,
)

prompt_template = """
### [INST] 
Instruction: Answer the question based on your 
gaming knowledge. 
If the answer cannot be found from the context, try to find the answer from your knowledge. 
If still unable to find the answer, respond with 'I don't know'.
Here is context to help:

{context}

### QUESTION:
{question} 

[/INST]
 """

llm = HuggingFacePipeline(pipeline=pipe)

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)
llm_chain = LLMChain(llm=llm, prompt=prompt)

print("load db")
vectorstore = load_vector_store()
retriever = vectorstore.as_retriever(
    search_kwargs={'k': 10, 'fetch_k': 50}
    )

# rag_chain = RetrievalQA.from_chain_type(llm,
#                                       chain_type='stuff',
#                                       retriever=retriever)

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | llm_chain
)


def ask(user_input):
    prompt = prompt_template.format(user_input=user_input)
    encoding = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        input_ids=encoding.input_ids,
        attention_mask=encoding.attention_mask,
        generation_config=generation_config
    )

    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resp


def ask_qa(user_input):
    resp = rag_chain.invoke(user_input)
    doc = resp['context'][0]
    page_content = doc.page_content
    source = doc.metadata['source']
    answer = resp['text']
    #answer = resp['result']
    print(f"{source=}")
    return answer


print(f"load {model_name} done")
with torch.inference_mode():
    while True:
        user_input = input("query: ")
        if user_input == '/bye':
            break
        if user_input == '':
            continue

        answer = ask_qa(user_input)
        print("--------------------------------------------------")
        print(answer)
        print("")
        print("--------------------------------------------------")
