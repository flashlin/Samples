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

from logger_lit import build_logger


class AppSettings:
    # The address of the model controller.
    controller_address: str = "http://localhost:21001"
    api_keys: Optional[List[str]] = None

app_settings = AppSettings()
get_bearer_token = HTTPBearer(auto_error=False)

async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if app_settings.api_keys:
        if auth is None or (token := auth.credentials) not in app_settings.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    # api_keys not set; allow all
    return None




"""
app = FastAPI()
@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    worker = worker_map[params["model"]]
    await acquire_worker_semaphore(worker)
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks(worker)
    return StreamingResponse(generator, background=background_tasks)
"""

