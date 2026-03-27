from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .types import Tool, create_tool

DEFAULT_MEMORY_ROOT = ".fury/memory"
DEFAULT_MEMORY_CHAR_LIMIT = 2500
ENTRY_DELIMITER = "\n§\n"

_THREAT_PATTERNS = [
    (r"ignore\s+(previous|all|above|prior)\s+instructions", "prompt_injection"),
    (r"system\s+prompt\s+override", "system_prompt_override"),
    (
        r"disregard\s+(your|all|any)\s+(instructions|rules|guidelines)",
        "disregard_rules",
    ),
    (r"do\s+not\s+tell\s+the\s+user", "deception_hide"),
    (
        r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)",
        "exfiltration",
    ),
    (r"authorized_keys", "ssh_backdoor"),
]
_INVISIBLE_CHARS = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
}

MEMORY_TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["add", "replace", "remove", "list"],
        },
        "content": {"type": "string"},
        "entry_id": {"type": "string"},
        "match_text": {"type": "string"},
        "pinned": {"type": "boolean"},
    },
    "required": ["action"],
}

MEMORY_TOOL_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "error": {"type": "string"},
        "message": {"type": "string"},
        "revision": {"type": "integer"},
        "scope": {"type": "object"},
        "store": {"type": "object"},
        "entry": {"type": "object"},
        "removed": {"type": "object"},
        "matches": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["success"],
}


@dataclass(frozen=True)
class MemoryScopeRef:
    key: str
    label: str


@dataclass(frozen=True)
class MemorySnapshot:
    revision: int
    scope: MemoryScopeRef
    content: str

    def render(self) -> str:
        return self.content


@dataclass
class MemoryEntry:
    id: str
    scope: str
    content: str
    created_at: str
    updated_at: str
    source: str = "agent"
    pinned: bool = False
    archived: bool = False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_memory_scope(scope: str) -> MemoryScopeRef:
    label = str(scope or "").strip()
    if not label:
        raise ValueError("memory scope must be a non-empty string")

    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", label).strip("-")
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:10]
    key = f"{slug or 'scope'}-{digest}"
    return MemoryScopeRef(key=key, label=label)


def compose_prompt_with_memory(
    base_prompt: str,
    snapshot: Optional[MemorySnapshot],
    *,
    heading: str = "Memory context:",
) -> str:
    prompt = base_prompt.rstrip()
    if snapshot is None:
        return prompt

    rendered = snapshot.render().strip()
    if not rendered:
        return prompt

    return f"{prompt}\n\n{heading}\n\n{rendered}"


def _scan_memory_content(content: str) -> Optional[str]:
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Blocked: content contains invisible unicode U+{ord(char):04X}."

    for pattern, label in _THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return f"Blocked: content matched safety rule '{label}'."

    return None


class MemoryStore:
    def __init__(
        self,
        root_dir: str = DEFAULT_MEMORY_ROOT,
        *,
        char_limit: int = DEFAULT_MEMORY_CHAR_LIMIT,
    ) -> None:
        self.root = Path(root_dir)
        self.scopes_dir = self.root / "scopes"
        self.char_limit = char_limit
        self._lock = threading.RLock()
        self._revision = 1
        self.root.mkdir(parents=True, exist_ok=True)
        self.scopes_dir.mkdir(parents=True, exist_ok=True)

    def capture_snapshot(self, scope: str) -> MemorySnapshot:
        scope_ref = resolve_memory_scope(scope)
        with self._lock:
            return MemorySnapshot(
                revision=self._revision,
                scope=scope_ref,
                content=self._render_prompt_block(scope_ref),
            )

    def get_scope(self, scope: str) -> Dict[str, Any]:
        scope_ref = resolve_memory_scope(scope)
        with self._lock:
            return self._serialize_store(scope_ref)

    def get_state(self, scope: str) -> Dict[str, Any]:
        scope_ref = resolve_memory_scope(scope)
        with self._lock:
            return {
                "revision": self._revision,
                "scope": asdict(scope_ref),
                "store": self._serialize_store(scope_ref),
            }

    def add(
        self,
        *,
        scope: str,
        content: str,
        source: str = "agent",
        pinned: bool = False,
    ) -> Dict[str, Any]:
        scope_ref = resolve_memory_scope(scope)
        with self._lock:
            result = self._add_entry_unlocked(
                scope_ref,
                content,
                source=source,
                pinned=pinned,
            )
            if result.get("success") and result.get("entry"):
                self._revision += 1
            return result

    def replace(
        self,
        *,
        scope: str,
        content: str,
        entry_id: Optional[str] = None,
        match_text: Optional[str] = None,
        pinned: Optional[bool] = None,
    ) -> Dict[str, Any]:
        scope_ref = resolve_memory_scope(scope)
        with self._lock:
            content = content.strip()
            if not content:
                return {"success": False, "error": "content is required."}

            scan_error = _scan_memory_content(content)
            if scan_error:
                return {"success": False, "error": scan_error}

            entries = self._read_entries(scope_ref)
            match = self._find_entry(entries, entry_id=entry_id, match_text=match_text)
            if "error" in match:
                return {
                    "success": False,
                    "error": match["error"],
                    "matches": match.get("matches"),
                }

            entry: MemoryEntry = match["entry"]
            updated_entries: List[MemoryEntry] = []
            for candidate in entries:
                if candidate.id != entry.id:
                    updated_entries.append(candidate)
                    continue
                candidate.content = content
                candidate.updated_at = utc_now_iso()
                if pinned is not None:
                    candidate.pinned = bool(pinned)
                updated_entries.append(candidate)

            budget_error = self._validate_budget(updated_entries)
            if budget_error:
                return {"success": False, "error": budget_error}

            self._write_entries(scope_ref, updated_entries)
            self._revision += 1
            return {
                "success": True,
                "entry": asdict(entry),
                "store": self._serialize_store(scope_ref),
            }

    def remove(
        self,
        *,
        scope: str,
        entry_id: Optional[str] = None,
        match_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        scope_ref = resolve_memory_scope(scope)
        with self._lock:
            entries = self._read_entries(scope_ref)
            match = self._find_entry(entries, entry_id=entry_id, match_text=match_text)
            if "error" in match:
                return {
                    "success": False,
                    "error": match["error"],
                    "matches": match.get("matches"),
                }

            entry: MemoryEntry = match["entry"]
            kept = [candidate for candidate in entries if candidate.id != entry.id]
            self._write_entries(scope_ref, kept)
            self._revision += 1
            return {
                "success": True,
                "removed": asdict(entry),
                "store": self._serialize_store(scope_ref),
            }

    def _add_entry_unlocked(
        self,
        scope_ref: MemoryScopeRef,
        content: str,
        *,
        source: str = "agent",
        pinned: bool = False,
    ) -> Dict[str, Any]:
        content = content.strip()
        if not content:
            return {"success": False, "error": "content is required."}

        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        entries = self._read_entries(scope_ref)
        if any(
            candidate.content == content and not candidate.archived
            for candidate in entries
        ):
            return {
                "success": True,
                "message": "Entry already exists.",
                "store": self._serialize_store(scope_ref),
            }

        now = utc_now_iso()
        entry = MemoryEntry(
            id=hashlib.sha1(
                f"{scope_ref.label}:{now}:{content}".encode("utf-8")
            ).hexdigest()[:12],
            scope=scope_ref.label,
            content=content,
            created_at=now,
            updated_at=now,
            source=source,
            pinned=pinned,
        )
        next_entries = [*entries, entry]
        budget_error = self._validate_budget(next_entries)
        if budget_error:
            return {"success": False, "error": budget_error}

        self._write_entries(scope_ref, next_entries)
        return {
            "success": True,
            "entry": asdict(entry),
            "store": self._serialize_store(scope_ref),
        }

    def _validate_budget(self, entries: Iterable[MemoryEntry]) -> Optional[str]:
        active_entries = [entry.content for entry in entries if not entry.archived]
        if not active_entries:
            return None

        total = len(ENTRY_DELIMITER.join(active_entries))
        if total <= self.char_limit:
            return None
        return (
            f"memory is {total}/{self.char_limit} chars; "
            "remove or consolidate entries first."
        )

    def _serialize_store(self, scope_ref: MemoryScopeRef) -> Dict[str, Any]:
        entries = self._sorted_entries(self._read_entries(scope_ref))
        active = [asdict(entry) for entry in entries if not entry.archived]
        usage = (
            len(ENTRY_DELIMITER.join(entry["content"] for entry in active))
            if active
            else 0
        )
        return {
            "scope": asdict(scope_ref),
            "entries": active,
            "usage_chars": usage,
            "limit_chars": self.char_limit,
            "usage_percent": int((usage / self.char_limit) * 100)
            if self.char_limit
            else 0,
        }

    def _render_prompt_block(self, scope_ref: MemoryScopeRef) -> str:
        store = self._serialize_store(scope_ref)
        entries = store["entries"]
        if not entries:
            return ""

        separator = "=" * 46
        body = ENTRY_DELIMITER.join(entry["content"] for entry in entries)
        return (
            f"{separator}\n"
            f"MEMORY SCOPE: {scope_ref.label} "
            f"[{store['usage_percent']}% - {store['usage_chars']}/{store['limit_chars']} chars]\n"
            f"{separator}\n"
            f"{body}"
        )

    def _find_entry(
        self,
        entries: List[MemoryEntry],
        *,
        entry_id: Optional[str],
        match_text: Optional[str],
    ) -> Dict[str, Any]:
        active = [entry for entry in entries if not entry.archived]

        if entry_id:
            for entry in active:
                if entry.id == entry_id:
                    return {"entry": entry}
            return {"error": f"No entry matched id '{entry_id}'."}

        needle = (match_text or "").strip()
        if not needle:
            return {"error": "entry_id or match_text is required."}

        matches = [entry for entry in active if needle.lower() in entry.content.lower()]
        if not matches:
            return {"error": f"No entry matched '{needle}'."}
        if len(matches) > 1:
            return {
                "error": f"Multiple entries matched '{needle}'. Be more specific.",
                "matches": [
                    {"id": entry.id, "preview": entry.content[:120]}
                    for entry in matches
                ],
            }
        return {"entry": matches[0]}

    def _path_for(self, scope_ref: MemoryScopeRef) -> Path:
        return self.scopes_dir / f"{scope_ref.key}.jsonl"

    def _read_entries(self, scope_ref: MemoryScopeRef) -> List[MemoryEntry]:
        path = self._path_for(scope_ref)
        if not path.exists():
            return []

        entries: List[MemoryEntry] = []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    try:
                        entries.append(
                            MemoryEntry(
                                id=str(payload.get("id", "")).strip(),
                                scope=str(payload.get("scope", scope_ref.label)).strip()
                                or scope_ref.label,
                                content=str(payload.get("content", "")).strip(),
                                created_at=str(payload.get("created_at", "")).strip()
                                or utc_now_iso(),
                                updated_at=str(payload.get("updated_at", "")).strip()
                                or utc_now_iso(),
                                source=str(payload.get("source", "agent")).strip()
                                or "agent",
                                pinned=bool(payload.get("pinned", False)),
                                archived=bool(payload.get("archived", False)),
                            )
                        )
                    except Exception:
                        continue
        except OSError:
            return []

        return [entry for entry in entries if entry.id and entry.content]

    def _write_entries(
        self, scope_ref: MemoryScopeRef, entries: List[MemoryEntry]
    ) -> None:
        path = self._path_for(scope_ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=".memory-",
            suffix=".jsonl",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for entry in self._sorted_entries(entries):
                    handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    @staticmethod
    def _sorted_entries(entries: List[MemoryEntry]) -> List[MemoryEntry]:
        return sorted(
            entries,
            key=lambda entry: (
                entry.pinned,
                entry.updated_at,
                entry.created_at,
            ),
            reverse=True,
        )


def create_memory_tool(
    store: MemoryStore,
    scope: str,
    *,
    name: str = "memory",
    description: Optional[str] = None,
) -> Tool:
    scope_ref = resolve_memory_scope(scope)

    def execute(
        action: str,
        content: Optional[str] = None,
        entry_id: Optional[str] = None,
        match_text: Optional[str] = None,
        pinned: Optional[bool] = None,
        emit: Optional[Callable[[Dict[str, str]], None]] = None,
    ) -> Dict[str, Any]:
        if emit is not None:
            verb = "Viewed" if action == "list" else "Updated"
            emit(
                {
                    "id": f"{name}:{action}:{scope_ref.key}",
                    "title": f"{verb} memory for {scope_ref.label}",
                    "type": "tool_call",
                }
            )

        if action == "add":
            result = store.add(
                scope=scope_ref.label,
                content=content or "",
                source="agent",
                pinned=bool(pinned),
            )
        elif action == "replace":
            result = store.replace(
                scope=scope_ref.label,
                content=content or "",
                entry_id=entry_id,
                match_text=match_text,
                pinned=pinned,
            )
        elif action == "remove":
            result = store.remove(
                scope=scope_ref.label,
                entry_id=entry_id,
                match_text=match_text,
            )
        elif action == "list":
            result = store.get_state(scope_ref.label)
            result["success"] = True
        else:
            result = {"success": False, "error": f"Unsupported action '{action}'."}

        if isinstance(result, dict):
            state = store.get_state(scope_ref.label)
            result.setdefault("revision", state.get("revision"))
            result.setdefault("scope", state.get("scope"))
            result.setdefault("store", state.get("store"))
        return result

    return create_tool(
        id=name,
        description=description
        or f"Store or update durable memory for the '{scope_ref.label}' scope.",
        execute=execute,
        input_schema=MEMORY_TOOL_INPUT_SCHEMA,
        output_schema=MEMORY_TOOL_OUTPUT_SCHEMA,
    )
