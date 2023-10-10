import time

from flask import Flask, request, Response, render_template, jsonify, redirect, url_for
from flask_cors import CORS, cross_origin
from typing import Any, List
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv

from llama2_utils import llama2_prompt
from model_utils import create_llama2

load_dotenv()
app = Flask(__name__)
cors = CORS(app)

class Item(BaseModel):
    prompt: str


llm = create_llama2()


@app.route('/api/v1/chat/completions', methods=['POST'])
def chat_completions():
    req = request.json
    messages = req['messages']
    print(f"{messages=}")
    resp = llm(llama2_prompt(messages))
    result = {
        'outputs': resp,
    }
    return jsonify(result)


@app.route('/api/v1/chat/stream', methods=['POST'])
def chat_stream():
    req = request.json
    messages = req['messages']
    resp = llm(llama2_prompt(messages))
    result = {
        'outputs': resp,
    }
    def generate():
        resp_len = len(resp)
        for idx in range(resp_len):
            token = resp[idx]
            yield token
            # time.sleep(0.1)

    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)
