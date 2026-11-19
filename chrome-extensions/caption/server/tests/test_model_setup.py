import sys
import types

from caption_server import model_setup
from caption_server.settings import Settings


def test_required_models_follow_server_settings() -> None:
    settings = Settings(_env_file=None, asr_model="org/asr", translate_model="org/mt")
    assert model_setup.required_models(settings) == ["org/asr", "org/mt"]


def test_required_models_are_deduplicated() -> None:
    settings = Settings(_env_file=None, asr_model="org/same", translate_model="org/same")
    assert model_setup.required_models(settings) == ["org/same"]


def test_main_downloads_each_model_without_xet(monkeypatch) -> None:
    downloaded: list[str] = []
    fake_hub = types.SimpleNamespace(snapshot_download=lambda repo_id: downloaded.append(repo_id) or f"/cache/{repo_id}")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.setenv("CAPTION_ASR_MODEL", "org/asr")
    monkeypatch.setenv("CAPTION_TRANSLATE_MODEL", "org/mt")
    model_setup.main()
    assert downloaded == ["org/asr", "org/mt"]
    import os

    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
