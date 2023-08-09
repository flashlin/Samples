import asyncio

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate

print("start")

# 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
print("tokenizer")
# bf16精度，A100、H100、RTX3060、RTX3070
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat",
#                                              device_map="auto",
#                                              bf16=True,
#                                              trust_remote_code=True).eval()

print("model")
# 可指定不同的生成长度、top_p等相关超参
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat",
                                                           trust_remote_code=True)

# -------------------------
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just sat that you don't know the answer,
don't try to make up an answer.

Context: {context}
Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer: 
"""


def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=[
        'context', 'question'])
    return prompt



def create_qa_pipe():
    print("create qa pipe")
    qa_prompt = set_custom_prompt()
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1024,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.95,
        repetition_penalty=1.2
    )
    print("local_llm")
    local_llm = HuggingFacePipeline(pipeline=pipe)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    DB_FAISS_PATH = "models/db_faiss"
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    print("db")
    qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
                                           chain_type="stuff",
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': qa_prompt}
                                           )
    qa_chain_fn = qa_chain(local_llm, qa_prompt, db)
    print("ready")
    return qa_chain_fn


async def bot_doc(qa_chain_fn, query):
    res = await qa_chain_fn.acall(query)
    answer = res["result"]
    sources = res["source_documents"]
    if sources:
        answer += f"\nSources:" + str(sources[0].metadata['source'])
    else:
        answer += f"\nNo Sources Found"
    print(f"{answer}")


def main_chat():
    history = None
    while True:
        user_input = input("query: ")
        response, history = model.chat(tokenizer, user_input, history=history)
        print(response)

async def main_doc():
    history = None
    while True:
        user_input = input("query: ")
        qa_chain_fn = create_qa_pipe()
        response = await bot_doc(qa_chain_fn, user_input)
        print(response)

if __name__ == '__main__':
    # asyncio.run(main())
    main_chat()
    


