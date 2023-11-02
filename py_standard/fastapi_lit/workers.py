import os
import time
import asyncio
import threading
from typing import Generator, Optional, Union, Dict, List, Any
from fastapi import Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from logger_lit import build_logger

WORKER_API_TIMEOUT = int(os.getenv("FASTAPI_WORKER_API_TIMEOUT", 100))
WORKER_HEART_BEAT_INTERVAL = int(os.getenv("FASTAPI_WORKER_HEART_BEAT_INTERVAL", 45))

class LlmToken(BaseModel):
    data: str

class BaseApiWorker:
    def __init__(
        self,
        worker_id: str,
        limit_worker_concurrency: int = 1,
    ):
        self.worker_id = worker_id
        self.limit_worker_concurrency = limit_worker_concurrency
        self.semaphore = None
        self.logger = build_logger("fastapi_worker", f"fastapi_worker_{self.worker_id}.log")

    def acquire_semaphore(self):
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.limit_worker_concurrency)
        return self.semaphore.acquire()

    def release_semaphore(self):
        self.semaphore.release()

    def generate_stream_gate(self, params):
        raise NotImplementedError

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
        generator = self.generate_stream_gate(params)
        background_tasks = BackgroundTasks()
        background_tasks.add_task(lambda: release_worker_semaphore(self))
        return StreamingResponse(generator, background=background_tasks, media_type="text/event-stream")
        # g = self.generate_stream_gate(params)
        # return StreamingResponse(g, media_type="text/event-stream")

    def get_queue_length(self):
        if (
            self.semaphore is None
            or self.semaphore._value is None
            or self.semaphore._waiters is None
        ):
            return 0
        else:
            return (
                self.limit_worker_concurrency
                - self.semaphore._value
                + len(self.semaphore._waiters)
            )

    def get_status(self):
        return {
            "queue_length": self.get_queue_length(),
        }


def release_worker_semaphore(worker: BaseApiWorker):
    worker.release_semaphore()





