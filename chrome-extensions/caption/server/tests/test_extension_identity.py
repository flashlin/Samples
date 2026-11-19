from pathlib import Path

from caption_server.extension_identity import unpacked_extension_id, unpacked_extension_origin
from caption_server.settings import Settings

CHROMIUM_OBSERVED_DIR = "/Users/user/vdisk/Samples/chrome-extensions/caption/dist"
CHROMIUM_OBSERVED_ID = "obeghcdepeodmjndcodofmbmbenpogpn"


def test_matches_the_id_chromium_assigned_to_an_unpacked_extension() -> None:
    assert unpacked_extension_id(CHROMIUM_OBSERVED_DIR) == CHROMIUM_OBSERVED_ID


def test_origin_uses_the_resolved_absolute_path(tmp_path: Path) -> None:
    dist = tmp_path / "caption" / "dist"
    dist.mkdir(parents=True)
    relative_like = dist / ".." / "dist"
    assert unpacked_extension_origin(relative_like) == f"chrome-extension://{unpacked_extension_id(str(dist.resolve()))}"


def test_dev_extension_dir_is_added_to_allowed_origins(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, allowed_origins=["chrome-extension://fixed"], dev_extension_dir=tmp_path)
    assert settings.effective_allowed_origins == ["chrome-extension://fixed", unpacked_extension_origin(tmp_path)]
    assert Settings(_env_file=None, allowed_origins=["chrome-extension://fixed"]).effective_allowed_origins == ["chrome-extension://fixed"]
