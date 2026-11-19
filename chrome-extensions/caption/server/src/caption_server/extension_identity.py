import hashlib
from pathlib import Path

_EXTENSION_ID_LENGTH = 32
_FIRST_ID_LETTER = ord("a")
_HEX_BASE = 16


def unpacked_extension_id(extension_dir: str) -> str:
    digest = hashlib.sha256(extension_dir.encode("utf-8")).hexdigest()[:_EXTENSION_ID_LENGTH]
    return "".join(chr(_FIRST_ID_LETTER + int(char, _HEX_BASE)) for char in digest)


def unpacked_extension_origin(extension_dir: Path) -> str:
    return f"chrome-extension://{unpacked_extension_id(str(extension_dir.resolve()))}"
