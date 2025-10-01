from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


class FileCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str) -> Optional[dict[str, Any]]:
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        path = self._key_to_path(key)
        try:
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
