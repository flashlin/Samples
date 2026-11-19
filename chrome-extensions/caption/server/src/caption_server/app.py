from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import JSONResponse

from .audio_decoder import AudioDecodeError, decode_to_mono_16k
from .caption_pipeline import CaptionPipeline
from .schemas import (
    PutSubtitleRequest,
    PutSubtitleResponse,
    SubtitleDocument,
    TranscriptionResponse,
    TranslationRequest,
    TranslationResponse,
)
from .security import RequestGuard
from .settings import Settings
from .subtitle_store import FileSubtitleStore, InvalidUrlKeyError, RevisionConflictError


def _read_limited(upload: UploadFile, limit: int) -> bytes:
    payload = upload.file.read(limit + 1)
    if len(payload) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"Audio exceeds {limit} bytes")
    if not payload:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Audio is empty")
    return payload


def _caption_routes(settings: Settings, pipeline: CaptionPipeline) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "engine": pipeline.engine_names}

    @router.post("/v1/captions/transcribe", response_model=TranscriptionResponse, response_model_by_alias=True)
    def transcribe(audio: UploadFile = File(...), translate: bool = Form(True)) -> TranscriptionResponse:
        try:
            samples = decode_to_mono_16k(_read_limited(audio, settings.max_audio_bytes))
        except AudioDecodeError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
        caption = pipeline.transcribe(samples, translate)
        return TranscriptionResponse(
            text=caption.text,
            language=caption.language.value,
            translation=caption.translation,
            display=caption.display,
            engine=pipeline.engine_names,
        )

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
        if len(request.srt.encode("utf-8")) > settings.max_srt_bytes:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"SRT exceeds {settings.max_srt_bytes} bytes")
        try:
            outcome = store.put(url_key, request.meta, request.srt, request.expected_revision, pipeline.engine_names)
        except InvalidUrlKeyError as error:
            raise invalid_key(error) from error
        except RevisionConflictError as conflict:
            body = PutSubtitleResponse(written=False, revision=conflict.current_revision, reason="revision-conflict")
            return JSONResponse(status_code=status.HTTP_409_CONFLICT, content=body.model_dump(by_alias=True))
        return PutSubtitleResponse(written=outcome.written, revision=outcome.revision, reason=outcome.reason)

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


def create_app(settings: Settings, pipeline: CaptionPipeline, store: FileSubtitleStore) -> FastAPI:
    app = FastAPI(title="caption-server", docs_url=None, redoc_url=None, openapi_url=None)
    guard = Depends(RequestGuard(settings))
    app.include_router(_caption_routes(settings, pipeline), dependencies=[guard])
    app.include_router(_subtitle_routes(settings, store, pipeline), dependencies=[guard])
    return app
