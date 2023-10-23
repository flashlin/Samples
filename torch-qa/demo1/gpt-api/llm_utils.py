import time
from dataclasses import dataclass
import json
import queue
from enum import Enum


class ResponseType(Enum):
    Stream = 1
    Message = 2


@dataclass
class ChatMessage:
    role: str
    content: str


class LLmTaskItem:
    login_name: str = ''
    conversation_id: int = 0
    messages: str = ''
    output_message: ChatMessage = ChatMessage(
        role='assistant',
        content=''
    )
    response_method: ResponseType = ResponseType.Stream
    is_finished: bool = False
    is_started: bool = False
    is_response_done: bool = False
    output_tokens = queue.Queue()

    def wait_for_start(self):
        while not self.is_started:
            time.sleep(0.5)

    def display(self, token: str):
        self.output_message.content += token
        self.output_tokens.put(token)
        self.is_started = True

    def display_end(self):
        self.is_finished = True

    def response_stream(self):
        print("=== response start ===")
        while not self.is_finished and self.output_tokens.not_empty:
            if not self.output_tokens.empty():
                output_token = self.output_tokens.get()
                if self.response_method == ResponseType.Stream:
                    json_str = json.dumps(output_token)
                    yield json_str + "\r\n"
                else:
                    yield output_token
                self.output_tokens.task_done()
                continue
            # time.sleep(0.2)
        if self.response_method == ResponseType.Stream:
            yield "data: [DONE]"
        self.is_response_done = True
        print("=== response end ===")

    def response(self):
        result = ""
        for token in self.response_stream():
            result += token
        return result

    def wait_for_response_done(self):
        while not self.is_response_done:
            time.sleep(0.5)
        time.sleep(0.5)


class LlmCallbackHandler:
    current_task_item: LLmTaskItem = None

    def display(self, text: str):
        self.current_task_item.display(text)

    def display_end(self):
        self.current_task_item.display_end()
