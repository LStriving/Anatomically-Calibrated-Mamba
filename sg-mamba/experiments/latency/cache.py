import hashlib
import json
import pickle
from pathlib import Path

def cache_key(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class StageCache:
    """Small disk cache for stage payloads; keys are controlled by the runner."""

    def __init__(self, root, enabled=True):
        self.root = Path(root)
        self.enabled = bool(enabled)
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key):
        return self.root / (key + ".pkl")

    def load(self, key):
        if not self.enabled:
            return None
        path = self.path_for(key)
        if not path.is_file():
            return None
        with path.open("rb") as handle:
            return pickle.load(handle)

    def save(self, key, payload):
        if not self.enabled:
            return None
        path = self.path_for(key)
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        return path
