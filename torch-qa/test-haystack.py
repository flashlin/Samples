from transformers import AutoModelForCausalLM, AutoTokenizer
from haystack.nodes import PromptNode, PromptModel
from haystack.agents.conversational import ConversationalAgent

MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
hf_token = ''

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_4bit=True, use_auth_token=hf_token)
# disable Tensor Parallelism (https://github.com/huggingface/transformers/pull/24906)
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=hf_token)

# inspiration: https://docs.haystack.deepset.ai/docs/prompt_node#using-models-not-supported-in-hugging-face-transformers
pn = PromptNode(MODEL_NAME,
                max_length=1000,
                model_kwargs={'model': model,
                              'tokenizer': tokenizer,
                              'task_name': 'text2text-generation',
                              'device': None,  # placeholder needed to make the underlying HF Pipeline work
                              'stream': True})

# quick sanity check
# input_text = "Describe the solar system."
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
# outputs = model.generate(input_ids, max_length=50)
# print(tokenizer.decode(outputs[0]))

# simply call the PromptNode
# pn("What's the coolest city in Italy? Explain reasons why")

prompt_template = """
[INST] <>
You are a helpful assistant who writes short answers.
<>\n\n
{memory} [INST] {query} [/INST]
"""

conversational_agent = ConversationalAgent(
    prompt_node=pn,
    prompt_template=prompt_template,
)
while True:
    query = input("\nHuman (type 'exit' or 'quit' to quit): ")
    if query.lower() == "exit" or query.lower() == "quit":
        break
    conversational_agent.run(query)
