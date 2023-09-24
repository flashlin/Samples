from typing import List, Union
import streamlit as st
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.callbacks.base import BaseCallbackHandler


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Reply your answer in markdown format.")
        ]


def get_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")


def convert_langchain_schema_to_dict(
        messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) \
        -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [
        {
            "role": get_role(message),
            "content": message.content
        } for message in messages
    ]


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
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method
        self.new_sentence = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.new_sentence += token

        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text + "▌")
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

    def on_llm_end(self, response, **kwargs) -> None:
        self.text = ""


def create_llama2(chat_box=None):
    if chat_box is None:
        chat_box = st.empty()
    display_handler = StreamDisplayHandler(chat_box)

    model_name = "CodeLlama-7B-Instruct-GGUF/codellama-7b-instruct.Q4_K_M.gguf"
    # model_name = "Llama-2-13B-chat-GGUF/llama-2-13b-chat.Q4_K_M.gguf"
    model_name = "Llama-2-7b-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler(), display_handler])
    return LlamaCpp(
        model_path=f"../models/{model_name}",
        #model_path='D:/Demo/qa-code/models/codellama-7b-instruct/codellama-7b-instruct.Q4_K_M.gguf',
        # input={
        #     "temperature": 0.1,
        #     "max_length": 2000,
        #     "top_p": 1
        # },
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        n_ctx=2048,  # 請求上下文 ValueError: Requested tokens (1130) exceed context window of 512
        callback_manager=callback_manager,
        verbose=False,  # True
        streaming=True,
    )
