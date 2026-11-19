import secrets

from .settings import Settings

_TOKEN_BYTES = 32
_OWNER_ONLY = 0o600


def _ensure_token(settings: Settings) -> Settings:
    if settings.token:
        return settings
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    if not settings.token_file.exists():
        settings.token_file.write_text(secrets.token_urlsafe(_TOKEN_BYTES), encoding="utf-8")
        settings.token_file.chmod(_OWNER_ONLY)
    return settings.model_copy(update={"token": settings.token_file.read_text(encoding="utf-8").strip()})


def _report_origins(settings: Settings) -> None:
    if settings.dev_extension_dir and not (settings.dev_extension_dir / "manifest.json").exists():
        print(f"Warning: {settings.dev_extension_dir} has no manifest.json, run `pnpm build` in caption/ first", flush=True)
    if settings.dev_extension_origin:
        print(f"Dev extension directory {settings.dev_extension_dir.resolve()} -> allowed origin {settings.dev_extension_origin}", flush=True)
    if not settings.effective_allowed_origins:
        print("Warning: no allowed origins configured, every request carrying an Origin header will be rejected", flush=True)


def main() -> None:
    import uvicorn

    from .app import create_app
    from .caption_pipeline import CaptionPipeline
    from .engines import HyMtTranslator, QwenAsrRecognizer
    from .subtitle_store import FileSubtitleStore

    settings = _ensure_token(Settings())
    _report_origins(settings)
    print(f"Bearer token file: {settings.token_file}", flush=True)
    print(f"Loading ASR model {settings.asr_model} and translation model {settings.translate_model}", flush=True)
    pipeline = CaptionPipeline(
        recognizer=QwenAsrRecognizer(settings.asr_model, settings.asr_language, settings.asr_device),
        translator=HyMtTranslator(settings.translate_model, settings.translate_max_tokens),
        keep_terms=settings.keep_terms,
        bilingual_english=settings.bilingual_english,
    )
    store = FileSubtitleStore(settings.subtitles_dir, settings.audit_log)
    uvicorn.run(create_app(settings, pipeline, store), host=settings.host, port=settings.port)
