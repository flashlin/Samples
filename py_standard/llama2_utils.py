from typing import List
from langchain.callbacks.base import BaseCallbackHandler
from typing import Callable


def llama2_prompt(messages: List[dict]) -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. 
    Always answer as helpfully as possible, while being safe. 
    Please ensure that your responses are socially unbiased and positive in nature. 
    If a question does not make any sense, or is not factually coherent, 
    explain why instead of answering something not correct. 
    If you don't know the answer to a question, please don't share false information."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages

    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(
        f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)


class StreamDisplayHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='display'):
        """
        :param container: class
        :param initial_text:
        :param display_method: 'display'
        class MyContainer:
            def display(self, text):
               pass
        """
        self.container = container
        self.display_method = display_method
        self.text = initial_text
        self.new_sentence = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.new_sentence += token

        # self.call_display_func(self.text + "▌")
        self.call_display_func(token)

    def on_llm_end(self, response, **kwargs) -> None:
        self.text = ""

    def call_display_func(self, text: str):
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(text)
        else:
            raise ValueError(f"Invalid display_method: {display_function}")
