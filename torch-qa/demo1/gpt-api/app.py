from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_cors import CORS, cross_origin
from celery.result import AsyncResult
from typing import Any
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
from celery_worker import generate_text_task
from torch.multiprocessing import Process, Queue


load_dotenv()
app = Flask(__name__)
cors = CORS(app)

class Item(BaseModel):
    prompt: str


@app.route('/api/v1/chat/completions', methods=['POST'])
def chat_completions():
    req = request.json
    result = {
        'result': 'Processed successfully',
        'data': req
    }
    return jsonify(result)


@app.post("/api/v1/chat/generate")
async def generate_text(item: Item) -> Any:
    task = generate_text_task.delay(item.prompt)
    return {"task_id": task.id}


@app.get("/api/v1/chat/task/{task_id}")
async def get_task(task_id: str) -> Any:
    result = AsyncResult(task_id)
    if result.ready():
        res = result.get()
        return {
            "isSuccess": True,
            "result": res[0],
            "time": res[1],
            "memory": res[2]
        }
    return {
        "isSuccess": False,
        "errorMessage": "Task not completed yet"
    }


if __name__ == '__main__':
    app.run(debug=True)
