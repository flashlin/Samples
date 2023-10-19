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

from mysql_utils import utc_time
from user_service import UserService
from flask_jwt_utils import create_auth_blueprint
from llama2_utils import llama2_prompt, GptMessage
from model_utils import create_llama2, create_llama2_v2
from dataclasses import dataclass
from langchain.callbacks.base import BaseCallbackHandler
from llm_utils import ChatMessage, TaskItem, LlmCallbackHandler
from mysql_gpt_repo_utils import MysqlGptRepo, AddConversationReq, AddConversationMessageReq
from gpt_service import GptService, GetConversationMessagesReq
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, JWTManager


load_dotenv()

llm_queue = queue.Queue(10)
llm_callback_handler = LlmCallbackHandler()
print(f"loading llm")
llm = create_llama2_v2(llm_callback_handler)
# llm = None
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
            # task_item.output_message = resp
            task_item.wait_for_response_done()
            gpt_db.add_conversation_detail(AddConversationMessageReq(
                ConversationId=task_item.conversation_id,
                RoleName='assistant',
                Message=task_item.output_message.content,
                CreateOn=utc_time()
            ))
            print(f"process end")


llm_task = LlmConsumer('consumer')
llm_task.daemon = True
llm_task.start()
print(f"consumer task started")


def convert_gpt_messages_to_dict_list(messages: list[GptMessage]):
    return [{
        'role': item.role,
        'content': item.content
        } for item in messages]


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


@app.route('/api/v1/chat/getLastConversation', methods=['POST'])
@cross_origin()
@jwt_required()
def chat_get_last_conversation():
    current_login_name = get_jwt_identity()
    gpt_service = GptService(gpt_db)
    resp = gpt_service.get_conversation_last_messages(current_login_name)
    return jsonify(resp)


@app.route('/api/v1/chat/cancelLastConversation', methods=['POST'])
@cross_origin()
@jwt_required()
def chat_cancel_last_conversation():
    current_login_name = get_jwt_identity()
    gpt_service = GptService(gpt_db)
    resp = gpt_service.cancel_last_conversation(current_login_name)
    return jsonify(resp)


@app.route('/api/v1/chat/conversation', methods=['POST'])
@cross_origin()
@jwt_required()
@stream_with_context
def chat_conversation():
    req = request.json
    conversation_id = req['conversationId']
    user_message = req['content']
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
    task_item.login_name = current_login_name
    task_item.conversation_id = conversation_id
    task_item.messages = llama2_prompt(convert_gpt_messages_to_dict_list(messages))
    print(f"{task_item.messages=}")
    llm_queue.put(task_item)
    task_item.wait_for_start()

    # def generate():
    #     for chunk in task_item.response():
    #         yield chunk
    #         try:
    #             # Check if the client has closed the connection
    #             request.environ.get('wsgi.input').read(1)
    #         except IOError:
    #             print("Client closed connection")
    #             break
    #
    # return Response(generate(), mimetype='text/event-stream')



    return Response(task_item.response(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.config['UserService'] = UserService(app, gpt_db)
    app.register_blueprint(create_auth_blueprint(app), url_prefix='/api/v1/auth')
    app.run(debug=True, threaded=True, use_reloader=False)
