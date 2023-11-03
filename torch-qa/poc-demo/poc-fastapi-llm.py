import asyncio
import json
import queue
import threading

from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi_lit.workers import FastApiWorkerBase, LlmToken
from langchain_lit import load_llm_model


class MyWorker(FastApiWorkerBase):
    count = 0
    def __init__(self):
        super().__init__(worker_id='id1')
        self.llm_tokens = queue.Queue()
        self.is_finished = False

    def display(self, text: str):
        self.llm_tokens.put(text)

    def display_end(self):
        self.is_finished = True

    async def process(self, params):
        global llm
        thread = threading.Thread(target=lambda: llm(params.message))
        thread.start()
        llm_tokens = self.llm_tokens
        while not self.is_finished:
            if llm_tokens.empty():
                await asyncio.sleep(0.9)
            else:
                token = llm_tokens.get()
                json_token = json.dumps(token)
                yield f"{json_token}\n\n"
        yield "data: [DONE]"

app = FastAPI()
worker = MyWorker()

llm = load_llm_model("./models/TheBloke_Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_M.gguf", worker)

class Req(BaseModel):
    message: str

@app.post("/test", response_model=LlmToken)
async def api_generate_stream(request: Req):
    return await worker.response_stream(request)

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=5005, log_level="info")