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

    from .app import CaptionServices, create_app
    from .caption_pipeline import CaptionPipeline, PipelineOptions
    from .engines import HyMtTranslator, QwenAsrRecognizer, SherpaPunctuator, SileroVadFactory, VadTuning
    from .subtitle_store import FileSubtitleStore, StorePolicy

    settings = _ensure_token(Settings())
    _report_origins(settings)
    print(f"Bearer token file: {settings.token_file}", flush=True)
    print(f"Loading ASR {settings.asr_model}, aligner {settings.aligner_model}, punctuation {settings.punctuation_model}, translation {settings.translate_model}", flush=True)
    pipeline = CaptionPipeline(
        recognizer=QwenAsrRecognizer(settings.asr_model, settings.asr_language, settings.asr_device, settings.aligner_model),
        punctuator=SherpaPunctuator(settings.punctuation_model),
        translator=HyMtTranslator(settings.translate_model, settings.translate_max_tokens),
        options=PipelineOptions(keep_terms=settings.keep_terms, bilingual_english=settings.bilingual_english, max_sentence_units=settings.max_sentence_units),
    )
    store = FileSubtitleStore(settings.subtitles_dir, settings.audit_log, StorePolicy(settings.history_versions, settings.completion_ratio))
    vad = SileroVadFactory(
        settings.vad_model,
        VadTuning(
            threshold=settings.vad_threshold,
            min_silence_seconds=settings.vad_min_silence_seconds,
            min_speech_seconds=settings.vad_min_speech_seconds,
            max_speech_seconds=settings.vad_max_speech_seconds,
        ),
    )
    app = create_app(settings, CaptionServices(pipeline=pipeline, store=store, new_detector=vad))
    uvicorn.run(app, host=settings.host, port=settings.port)
