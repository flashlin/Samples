import json
import os

from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_cors import CORS, cross_origin
from typing import Any
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
from torch.multiprocessing import Process, Queue
import torch.distributed as dist
from model_utils import create_llama2

WORLD_SIZE = 2


request_queues = [Queue() for _ in range(WORLD_SIZE)]
response_queues = [Queue() for _ in range(WORLD_SIZE)]


def run(rank, request_queue, response_queue):
    os.environ['LOCAL_RANK'] = str(rank)
    generator = create_llama2()
    # send initialization signal
    response_queue.put("INITIALIZED")

    while True:
        # load messages from queue
        dialogs = [request_queue.get()]

        # replace Llama 2 default system message
        if dialogs[0][0]["role"] != "system":
            dialogs[0] = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant",
                }
            ] + dialogs[0]

        # send messages to Llama 2
        response = generator.generate(dialogs)
        response_queue.put(response)


def init_process(rank, fn, request_queue, response_queue, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str('5000')
    os.environ['WORLD_SIZE'] = str(WORLD_SIZE)
    dist.init_process_group(backend, rank=rank, world_size=WORLD_SIZE)
    fn(rank, request_queue, response_queue)



def respond_json(response, key="message"):
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "choices": [{
            "index": 0,
            key: response,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }


def check_messages(messages):
    if not isinstance(messages, list):
        return jsonify({
            "errorMessage": "'messages' must be a list",
        }), 400

    for message in messages:
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            return jsonify({
               "errorMessage": "Each message must have a 'role' and a 'content'",
            }), 400

    return None



def message_route():
    # get messages from request
    messages = request.json.get("messages")

    # validate message format
    errors = check_messages(messages)
    if errors:
        return errors

    # add messages to queue for Llama 2
    for rank in range(WORLD_SIZE):
        request_queues[rank].put(messages)

    # wait for response
    for rank in range(WORLD_SIZE):
        response = response_queues[rank].get()

    # return mocked stream response
    if request.json.get("stream"):
        maxlen = 128
        rc = response["content"]
        deltas = [rc[i:i+maxlen] for i in range(0, len(rc), maxlen)]
        output = ""
        for delta in deltas:
            delta_response = {
                "role": response["role"],
                "content": delta
            }
            output += "data: " + json.dumps(respond_json(delta_response, "delta")) + "\n"
        return output + "data: [DONE]"

    # return regular JSON response
    return jsonify(respond_json(response))




def startup():
    processes = []
    # initialize all Llama 2 processes
    for rank in range(WORLD_SIZE):
        p = Process(target=init_process, args=(rank, run, request_queues[rank], response_queues[rank]))
        p.start()
        processes.append(p)

    # wait for Llama 2 initialization
    for rank in range(WORLD_SIZE):
        response = response_queues[rank].get()

    print("\nStarting Flask API...")
    load_dotenv()
    app = Flask(__name__)
    cors = CORS(app)

    app.route("/chat", methods=["POST"])(message_route)
    app.run(host='0.0.0.0', port=5005)

    for p in processes:
        p.join()



class Item(BaseModel):
    prompt: str


if __name__ == '__main__':
    startup()
