import asyncio
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi_lit.workers import FastApiWorkerBase, LlmToken


class MyWorker(FastApiWorkerBase):
    count = 0

    def __init__(self):
        super().__init__(worker_id='id1')

    async def process(self, params):
        await asyncio.sleep(3)
        self.count += 1
        yield f"data: {self.count}\n\n"
        print("END")


app = FastAPI()
worker = MyWorker()


class Req(BaseModel):
    message: str

@app.post("/test", response_model=LlmToken)
async def api_generate_stream(request: Req):
    return await worker.response_stream(request)

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=5005, log_level="info")