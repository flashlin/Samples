import time
from dataclasses import dataclass
import json
import queue


@dataclass
class ChatMessage:
    role: str
    content: str


class TaskItem:
    messages: str = ''
    output_message: ChatMessage = ChatMessage(
        role='user',
        content=''
    )
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

    def response(self):
        print("=== response start ===")
        while not self.is_finished and self.output_tokens.not_empty:
            if self.output_tokens.not_empty:
                output_token = self.output_tokens.get()
                yield json.dumps(output_token) + "\r\n"
                self.output_tokens.task_done()
                continue
            time.sleep(0.2)
        yield "data: [DONE]"
        self.is_response_done = True
        print("=== response end ===")

    def wait_for_response_done(self):
        while not self.is_response_done:
            time.sleep(0.5)
        time.sleep(0.5)


class LlmCallbackHandler:
    current_task_item: TaskItem = None

    def display(self, text: str):
        self.current_task_item.display(text)

    def display_end(self):
        self.current_task_item.display_end()