import json
import os
import time
import queue
from flask import Flask, request, Response, render_template, jsonify, redirect, url_for, stream_with_context, \
    current_app
from flask_cors import CORS, cross_origin
from typing import Any, List
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
import threading

from user_service import UserService
from flask_jwt_utils import create_auth_blueprint
from llama2_utils import llama2_prompt
from model_utils import create_llama2, create_llama2_v2
from dataclasses import dataclass
from langchain.callbacks.base import BaseCallbackHandler
from llm_utils import ChatMessage, TaskItem, LlmCallbackHandler
from mysql_gpt_repo_utils import MysqlGptRepo, AddConversationReq
from gpt_service import GptService, GetConversationMessagesReq
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, JWTManager


load_dotenv()

llm_queue = queue.Queue(10)
llm_callback_handler = LlmCallbackHandler()
print(f"loading llm")
llm = create_llama2_v2(llm_callback_handler)
gpt_db = MysqlGptRepo()


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
            print(f"start process")
            resp = llm(task_item.messages)
            task_item.output_message = resp
            task_item.wait_for_response_done()


llm_task = LlmConsumer('consumer')
llm_task.daemon = True
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
    task_item.wait_for_start()
    return Response(stream_with_context(task_item.response()), mimetype='text/event-stream')


@app.route('/api/v1/chat/conversation', methods=['POST'])
@cross_origin()
@jwt_required()
def chat_conversation():
    req = request.json
    conversation_id = req['conversationId']
    user_message = req['message']
    current_login_name = get_jwt_identity()
    gpt_service = GptService(gpt_db)
    gpt_service.add_conversation_message(AddConversationReq(
        conversation_id=conversation_id,
        login_name=current_login_name,
        message=user_message
    ))
    messages = gpt_service.get_conversation_messages(GetConversationMessagesReq(
        conversation_id=conversation_id,
        login_name=current_login_name
    ))
    task_item = TaskItem()
    task_item.messages = llama2_prompt(messages)
    print(f"{task_item.messages=}")
    llm_queue.put(task_item)
    task_item.wait_for_start()
    return Response(stream_with_context(task_item.response()), mimetype='text/event-stream')


if __name__ == '__main__':
    app.config['UserService'] = UserService(app, gpt_db)
    app.register_blueprint(create_auth_blueprint(app), url_prefix='/api/v1/auth')
    app.run(debug=True, threaded=True, use_reloader=False)
