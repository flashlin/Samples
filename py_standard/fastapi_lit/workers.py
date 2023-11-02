import os
from fastapi.responses import StreamingResponse
from fastapi import BackgroundTasks
from pydantic import BaseModel

from thread_lit import ThreadWorkerBase

WORKER_API_TIMEOUT = int(os.getenv("FASTAPI_WORKER_API_TIMEOUT", 100))
WORKER_HEART_BEAT_INTERVAL = int(os.getenv("FASTAPI_WORKER_HEART_BEAT_INTERVAL", 45))


class LlmToken(BaseModel):
    data: str


class FastApiWorkerBase(ThreadWorkerBase):

    async def response_stream(self, params):
        """
            @app.post("/worker_generate_stream")
            async def api_generate_stream(request: Request):
                params = await request.json()
                worker = worker_map[params["model"]]
                await worker.acquire_semaphore(worker)
                return worker.response_stream()
        """
        await self.acquire_semaphore()
        generator = self.process(params)
        background_tasks = BackgroundTasks()
        background_tasks.add_task(lambda: self.release_semaphore())
        return StreamingResponse(generator, background=background_tasks, media_type="text/event-stream")



