import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional


DEFAULT_MEMORY_PATH = os.path.expanduser("~/.smolagents/memory.json")


@dataclass
class MemoryRecord:
    id: int
    text: str
    created_at: str
    tags: List[str]
    source: Optional[str] = None


class PersistentMemory:
    """Long term memory that persists across agent runs.

    Args:
        path: where the memory file lives. Defaults to ~/.smolagents/memory.json
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or DEFAULT_MEMORY_PATH
        self._records: List[MemoryRecord] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self._records = [MemoryRecord(**r) for r in data]
            except Exception:
                # file is probably corrupted, start fresh
                self._records = []

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump([asdict(r) for r in self._records], f, indent=2)

    def remember(self, text: str, tags: List[str] = [], source: str = None) -> MemoryRecord:
        """Store a new memory and persist it to disk."""
        record = MemoryRecord(
            id=len(self._records),
            text=text,
            created_at=str(datetime.now()),
            tags=tags,
            source=source,
        )
        self._records.append(record)
        self._save()
        return record

    def recall(self, query: str, top_k: int = 5, tags: List[str] = []) -> List[MemoryRecord]:
        """Return the top_k memories most relevant to the query."""
        query_words = set(query.lower().split())
        scored = []
        for r in self._records:
            if tags and not set(tags) & set(r.tags):
                continue
            words = set(r.text.lower().split())
            score = len(query_words & words)
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        # print(scored)
        return [r for s, r in scored][:top_k]

    def forget(self, id: int):
        self._records = [r for r in self._records if r.id != id]
        self._save()

    def all(self) -> List[MemoryRecord]:
        return self._records

    def clear(self):
        self._records = []
        self._save()

    def __len__(self):
        return len(self._records)


_default_store = None


def get_default_memory_store() -> PersistentMemory:
    global _default_store
    if _default_store is None:
        _default_store = PersistentMemory()
    return _default_store
