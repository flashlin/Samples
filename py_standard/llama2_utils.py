from dataclasses import dataclass
from datetime import datetime

DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. 
Always answer as helpfully as possible, while being safe. 
Please ensure that your responses are socially unbiased and positive in nature. 
If a question does not make any sense, or is not factually coherent, 
explain why instead of answering something not correct. 
If you don't know the answer to a question, please don't share false information."""


@dataclass
class GptMessage:
    id: int
    role: str
    content: str
    create_on: datetime


def convert_gpt_messages_to_dict_list(messages: list[GptMessage]):
    return [{
        'role': item.role,
        'content': item.content
    } for item in messages]


def llama2_prompt(messages: list[dict]) -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"

    if messages[0]['role'] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages

    messages = [
        {
            "role": messages[1]['role'],
            "content": B_SYS + messages[0]['content'] + E_SYS + messages[1]['content'],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]

    messages_list.append(
        f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)


