"""Personal assistant on the node — local LLM with a JSON memory file and
live context timeline (thothcraft earbuds/voice feature).

Wraps :class:`~thoth.ai_services.AIServices.chat` with:

* **Memory** — ``memory.json`` in the AI state dir::

      {
        "facts": {"<key>": "<value>"},        # durable user facts
        "people": {"<name>": "<note>"},       # who is who
        "preferences": {"<key>": "<value>"},  # how the user wants things
        "summary": "<rolling paragraph>",     # what happened lately
        "history": [{"role", "text", "ts"}]   # bounded chat log
      }

  ``history`` appends after every turn; the ``facts``/``people``/
  ``preferences``/``summary`` fields are updated by a second, cheap
  "reflect" pass at the end of each chat (``settings.reflect``), which
  asks the model for a compact JSON patch and merges it best-effort.

* **Context timeline** — a ``context_fn`` supplied by the daemon returns
  recent node events (presence/zone/activity predictions, gestures,
  digital-activity samples) inside ``settings.history_s``; they go into
  the system prompt so the assistant can answer "what was I doing" and
  notice the room around it.

* **Settings** — stored inside the same memory doc under ``settings`` so
  GET/PUT is a single file:

      history_s (float)  how far back the context timeline reaches
      max_tokens (int)   reply budget
      temperature (float)
      reflect (bool)     post-chat memory update pass
      persona (str)      system-prompt persona line

Every surface degrades when the LLM isn't configured — ``chat`` returns
``{"status": "unavailable", ...}`` like ``AIServices.chat``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_SETTINGS: Dict[str, Any] = {
    "history_s": 3600.0,
    "max_tokens": 300,
    "temperature": 0.6,
    "reflect": True,
    "persona": "You are Thoth, a private assistant living on Room1. "
               "Be brief, warm, and concrete — replies are often spoken "
               "through earbuds.",
}
_HISTORY_MAX = 100


def _default_memory() -> Dict[str, Any]:
    return {"facts": {}, "people": {}, "preferences": {},
            "summary": "", "history": [], "settings": dict(_DEFAULT_SETTINGS)}


class Assistant:
    """Owns memory.json + builds the chat context; one per daemon."""

    def __init__(self, ai: Any, memory_path: Path,
                 context_fn: Optional[Callable[[float], List[Dict[str, Any]]]] = None):
        self._ai = ai
        self.path = Path(memory_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._context_fn = context_fn or (lambda _s: [])
        self._lock = threading.RLock()

    # -- memory -------------------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        try:
            doc = json.loads(self.path.read_text())
        except Exception:
            doc = {}
        base = _default_memory()
        for k in base:
            if k in doc and isinstance(doc[k], type(base[k])):
                base[k] = doc[k]
        return base

    def _save(self, mem: Dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(mem, indent=2, ensure_ascii=False))
        tmp.replace(self.path)

    def memory(self) -> Dict[str, Any]:
        with self._lock:
            return self._load()

    def put_memory(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """App-side edit: shallow-merge facts/people/preferences, replace
        summary, merge settings. History is append-only via chat."""
        with self._lock:
            mem = self._load()
            for k in ("facts", "people", "preferences"):
                v = patch.get(k)
                if isinstance(v, dict):
                    for key, val in v.items():
                        if val is None:
                            mem[k].pop(key, None)
                        else:
                            mem[k][key] = val
            if isinstance(patch.get("summary"), str):
                mem["summary"] = patch["summary"]
            if isinstance(patch.get("settings"), dict):
                self._merge_settings(mem, patch["settings"])
            self._save(mem)
            return mem

    def clear_memory(self) -> Dict[str, Any]:
        with self._lock:
            mem = _default_memory()
            self._save(mem)
            return mem

    @staticmethod
    def _merge_settings(mem: Dict[str, Any], patch: Dict[str, Any]) -> None:
        st = mem.setdefault("settings", dict(_DEFAULT_SETTINGS))
        for k, default in _DEFAULT_SETTINGS.items():
            if k in patch:
                try:
                    st[k] = type(default)(patch[k])
                except (TypeError, ValueError):
                    st[k] = patch[k]

    # -- context --------------------------------------------------------------
    def _timeline_text(self, mem: Dict[str, Any]) -> str:
        st = mem["settings"]
        try:
            events = self._context_fn(float(st.get("history_s", 3600.0))) or []
        except Exception as exc:
            logger.debug("assistant context failed: %s", exc)
            return ""
        if not events:
            return ""
        lines = []
        for ev in events[-60:]:
            ts = ev.get("ts") or ev.get("timestamp")
            clock = time.strftime("%H:%M", time.localtime(float(ts))) \
                if isinstance(ts, (int, float)) else "?"
            kind = ev.get("kind") or ev.get("task") or "event"
            label = ev.get("label") or ev.get("value") or ""
            dev = ev.get("device") or ev.get("subject") or ""
            lines.append(f"[{clock}] {kind} {dev} {label}".rstrip())
        return "Recent events:\n" + "\n".join(lines)

    def _system_prompt(self, mem: Dict[str, Any]) -> str:
        st = mem["settings"]
        parts = [str(st.get("persona") or _DEFAULT_SETTINGS["persona"])]
        if mem.get("summary"):
            parts.append(f"What you know so far: {mem['summary']}")
        facts = {**(mem.get("facts") or {}), **(mem.get("preferences") or {})}
        if facts:
            parts.append("User facts: " + "; ".join(
                f"{k}={v}" for k, v in list(facts.items())[:40]))
        if mem.get("people"):
            parts.append("People: " + "; ".join(
                f"{k} — {v}" for k, v in list(mem["people"].items())[:20]))
        tl = self._timeline_text(mem)
        if tl:
            parts.append(tl)
        return "\n\n".join(parts)

    # -- chat ------------------------------------------------------------------
    def chat(self, text: str, max_tokens: Optional[int] = None,
             reflect: Optional[bool] = None) -> Dict[str, Any]:
        """One turn: memory+timeline context → LLM → history + memory update."""
        text = (text or "").strip()
        if not text:
            return {"status": "failed", "error": "empty text"}
        with self._lock:
            mem = self._load()
            st = mem["settings"]
            messages = [{"role": "system", "content": self._system_prompt(mem)}]
            messages += [{"role": h["role"], "content": h["text"]}
                         for h in mem["history"][-20:]
                         if h.get("role") in ("user", "assistant")]
            messages.append({"role": "user", "content": text})
        out = self._ai.chat(
            messages,
            max_tokens=int(max_tokens if max_tokens is not None
                           else st.get("max_tokens", 300)),
            temperature=float(st.get("temperature", 0.6)))
        if out.get("status") != "ok":
            return out
        reply = str(out.get("text") or "").strip()
        with self._lock:
            mem = self._load()
            mem["history"].append({"role": "user", "text": text, "ts": time.time()})
            mem["history"].append({"role": "assistant", "text": reply, "ts": time.time()})
            mem["history"] = mem["history"][-_HISTORY_MAX:]
            self._save(mem)
        if reflect if reflect is not None else bool(st.get("reflect", True)):
            try:
                self._reflect(text, reply)
            except Exception as exc:
                logger.debug("assistant reflect failed: %s", exc)
        return {"status": "ok", "text": reply, "model": out.get("model")}

    def _reflect(self, user_text: str, reply: str) -> None:
        """Post-chat memory pass — one small completion producing a JSON
        patch merged into facts/people/preferences/summary."""
        mem = self._load()
        prompt = (
            "Update the assistant's JSON memory after this exchange.\n"
            f"USER: {user_text}\nASSISTANT: {reply}\n\n"
            "Current memory:\n" + json.dumps(
                {"facts": mem["facts"], "people": mem["people"],
                 "preferences": mem["preferences"], "summary": mem["summary"]},
                ensure_ascii=False) + "\n\n"
            'Reply with ONLY a JSON object: {"facts": {"k": "v or null to '
            'delete"}, "people": {}, "preferences": {}, "summary": '
            '"<=40 words"} — include only keys that changed; {} is fine.')
        out = self._ai.chat(
            [{"role": "system", "content": "You output JSON only."},
             {"role": "user", "content": prompt}],
            max_tokens=220, temperature=0.0)
        if out.get("status") != "ok":
            return
        raw = str(out.get("text") or "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return
        try:
            patch = json.loads(raw[start:end + 1])
        except Exception:
            return
        if isinstance(patch, dict):
            self.put_memory(patch)


__all__ = ["Assistant"]
