import asyncio
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from fastapi import FastAPI
from flask import Request
from pydantic import BaseModel

from fastapi_lit.workers import BaseApiWorker, LlmToken, create_background_tasks


class MyWorker(BaseApiWorker):
    count = 0

    def __init__(self):
        super().__init__(worker_id='id1')

    def generate_stream_gate(self, params):
        asyncio.sleep(1000 * 10)
        self.count += 1
        yield f"data: {self.count}\n\n"
        print("END")


app = FastAPI()
worker = MyWorker()


class Req(BaseModel):
    message: str

@app.post("/test", response_model=LlmToken)
async def api_generate_stream(request: Req):
    await worker.acquire_semaphore()
    # return worker.response_stream(request)
    generator = worker.generate_stream_gate(request)
    background_tasks = create_background_tasks(worker)
    return StreamingResponse(generator, background=background_tasks, media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=5005, log_level="info")