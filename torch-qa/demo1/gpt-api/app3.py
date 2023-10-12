import time
import queue
from flask import Flask, request, Response, render_template, jsonify, redirect, url_for
from flask_cors import CORS, cross_origin
from typing import Any, List
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
import threading
from llama2_utils import llama2_prompt
from model_utils import create_llama2, create_llama2_v2
from dataclasses import dataclass
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()


class Item(BaseModel):
    prompt: str


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
    output_tokens = queue.Queue()

    def display(self, token: str):
        self.output_message.content += token
        self.output_tokens.put(token)

    def response(self):
        print("=== response start ===")
        while not self.is_finished and self.output_tokens.not_empty:
            if self.output_tokens.not_empty:
                output_token = self.output_tokens.get()
                yield output_token
                self.output_tokens.task_done()
                continue
            print("---- waiting ---")
            time.sleep(0.5)
        yield "\r\ndata: [DONE]\r\n"
        print("=== response end ===")


class LlmCallbackHandler:
    current_task_item: TaskItem = None

    def display(self, text: str):
        self.current_task_item.display(text)


llm_queue = queue.Queue(10)
llm_callback_handler = LlmCallbackHandler()
print(f"loading llm")
llm = create_llama2_v2(llm_callback_handler)


class LlmConsumer(threading.Thread):
    def __init__(self, thread_name):
        super(LlmConsumer, self).__init__(name=thread_name)

    def run(self):
        global llm_queue
        while True:
            if llm_queue.empty():
                time.sleep(1)
                continue
            task_item: TaskItem = llm_queue.get()
            llm_callback_handler.current_task_item = task_item
            resp = llm(task_item.messages)
            task_item.output_message = resp
            task_item.is_finished = True
            llm_queue.task_done()


llm_task = LlmConsumer('consumer')
llm_task.start()
print(f"consumer task started")

app = Flask(__name__)
cors = CORS(app)



@app.route('/api/v1/chat/completions', methods=['POST'])
def chat_completions():
    req = request.json
    messages = req['messages']
    resp = llm(llama2_prompt(messages))
    result = {
        'outputs': resp,
    }
    return jsonify(result)


@app.route('/api/v1/chat/stream', methods=['POST'])
def chat_stream():
    req = request.json
    messages = req['messages']
    task_item = TaskItem()
    task_item.messages = llama2_prompt(messages)
    llm_queue.put(task_item)
    return Response(task_item.response(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
