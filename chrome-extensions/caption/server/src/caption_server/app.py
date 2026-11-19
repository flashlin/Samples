from collections.abc import Callable
from dataclasses import dataclass

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Response, WebSocket, status
from fastapi.responses import JSONResponse

from .caption_pipeline import CaptionPipeline
from .schemas import (
    PutSubtitleRequest,
    PutSubtitleResponse,
    SubtitleDocument,
    UndoResponse,
    TranslationRequest,
    TranslationResponse,
)
from .security import RequestGuard
from .settings import Settings
from .srt_codec import Cue
from .stream_endpoint import pipeline_engines, serve_caption_stream
from .stream_session import SpeechDetector, StreamTuning
from .subtitle_store import FileSubtitleStore, InvalidUrlKeyError, RevisionConflictError, SubtitleWrite


@dataclass(frozen=True)
class CaptionServices:
    pipeline: CaptionPipeline
    store: FileSubtitleStore
    new_detector: Callable[[], SpeechDetector]


def _caption_routes(settings: Settings, pipeline: CaptionPipeline) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "engine": pipeline.engine_names}

    @router.post("/v1/translations", response_model=TranslationResponse, response_model_by_alias=True)
    def translate(request: TranslationRequest) -> TranslationResponse:
        if len(request.texts) > settings.max_translation_lines:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"At most {settings.max_translation_lines} lines per request")
        captions = [pipeline.caption_text(text) for text in request.texts]
        return TranslationResponse(
            translations=[caption.translation or caption.text for caption in captions],
            displays=[caption.display for caption in captions],
        )

    return router


def _subtitle_routes(settings: Settings, store: FileSubtitleStore, pipeline: CaptionPipeline) -> APIRouter:
    router = APIRouter(prefix="/v1/subtitles/{url_key}")

    def invalid_key(error: InvalidUrlKeyError) -> HTTPException:
        return HTTPException(status.HTTP_400_BAD_REQUEST, str(error))

    @router.get("", response_model=SubtitleDocument, response_model_by_alias=True)
    def get_subtitle(url_key: str) -> SubtitleDocument:
        try:
            document = store.get(url_key)
        except InvalidUrlKeyError as error:
            raise invalid_key(error) from error
        if document is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Subtitle not found")
        return document

    @router.put("", response_model=PutSubtitleResponse, response_model_by_alias=True)
    def put_subtitle(url_key: str, request: PutSubtitleRequest) -> PutSubtitleResponse | JSONResponse:
        try:
            write = SubtitleWrite(
                meta=request.meta,
                cues=[Cue(cue.start, cue.end, cue.text) for cue in request.cues],
                placement=request.placement,
                expected_revision=request.expected_revision,
                confirm_overwrite=request.confirm_overwrite,
                engine=pipeline.engine_names,
            )
            outcome = store.put(url_key, write)
        except InvalidUrlKeyError as error:
            raise invalid_key(error) from error
        except RevisionConflictError as conflict:
            body = PutSubtitleResponse(written=False, revision=conflict.current_revision, reason="revision-conflict")
            return JSONResponse(status_code=status.HTTP_409_CONFLICT, content=body.model_dump(by_alias=True))
        return PutSubtitleResponse(written=outcome.written, revision=outcome.revision, reason=outcome.reason)

    @router.post("/undo", response_model=UndoResponse, response_model_by_alias=True)
    def undo_subtitle(url_key: str) -> UndoResponse:
        try:
            outcome = store.undo(url_key)
        except InvalidUrlKeyError as error:
            raise invalid_key(error) from error
        return UndoResponse(restored=outcome.restored, revision=outcome.revision, remaining_versions=outcome.remaining_versions)

    @router.delete("", status_code=status.HTTP_204_NO_CONTENT)
    def delete_subtitle(url_key: str) -> Response:
        try:
            deleted = store.delete(url_key)
        except InvalidUrlKeyError as error:
            raise invalid_key(error) from error
        if not deleted:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Subtitle not found")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router


def create_app(settings: Settings, services: CaptionServices) -> FastAPI:
    app = FastAPI(title="caption-server", docs_url=None, redoc_url=None, openapi_url=None)
    guard = RequestGuard(settings)
    app.include_router(_caption_routes(settings, services.pipeline), dependencies=[Depends(guard)])
    app.include_router(_subtitle_routes(settings, services.store, services.pipeline), dependencies=[Depends(guard)])
    engines = pipeline_engines(services.pipeline, services.new_detector)
    tuning = StreamTuning(partial_interval_seconds=settings.stream_partial_interval_seconds)

    @app.websocket("/v1/captions/stream")
    async def caption_stream(websocket: WebSocket) -> None:
        await serve_caption_stream(websocket, guard, engines, tuning)

    return app
