from flask import Flask, request, render_template, jsonify, redirect, url_for
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
    print(f"{resp=}")
    result = {
        'outputs': resp,
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
