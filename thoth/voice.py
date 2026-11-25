"""Voice session orchestration — earbuds double-tap → spoken chat with the
local assistant (thothcraft buds feature).

Flow: the buds raise a ``thoth_voice_toggle`` gesture → the daemon opens a
:class:`VoiceSession`, pushes ``listening`` session state back to the buds,
then the client (app/desktop) POSTs audio turns to ``/api/v1/voice/turn``:

    wav bytes → AIServices.stt → Assistant.chat → AIServices.tts
    → {transcript, reply, audio_b64}  (+ session state feedback)

The session object tracks its open/turn counters so the API can report
state; audio I/O stays with the client (the phone plays the TTS reply
through the buds' classic-BT audio link).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_IDLE_TIMEOUT_S = 120.0


class VoiceSession:
    """One live voice-chat session on the node."""

    def __init__(self, ai: Any, assistant: Any,
                 feedback: Optional[Callable[[str], Any]] = None,
                 emit_event: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
                 device_id: str = ""):
        self._ai = ai
        self._assistant = assistant
        self._feedback = feedback          # state -> buds session char
        self._emit = emit_event            # node event fanout
        self.device_id = device_id
        self.opened_at = time.time()
        self.last_activity = self.opened_at
        self.turns = 0
        self._set_state("listening")

    # -- state -----------------------------------------------------------------
    def _set_state(self, state: str) -> None:
        if self._feedback:
            try:
                self._feedback(state)
            except Exception as exc:
                logger.debug("buds session feedback failed: %s", exc)

    def _event(self, kind: str, **data) -> None:
        if self._emit:
            try:
                self._emit(kind, {"device_id": self.device_id, "ts": time.time(), **data})
            except Exception:
                pass

    def expired(self, now: Optional[float] = None) -> bool:
        return (now or time.time()) - self.last_activity > _IDLE_TIMEOUT_S

    def status(self) -> Dict[str, Any]:
        return {"device_id": self.device_id, "opened_at": self.opened_at,
                "last_activity": self.last_activity, "turns": self.turns}

    # -- turns -------------------------------------------------------------------
    def turn(self, wav_bytes: bytes, language: Optional[str] = None,
             speak: bool = True) -> Dict[str, Any]:
        """One turn: wav → transcript → assistant reply → optional TTS audio."""
        self.last_activity = time.time()
        self._set_state("listening")
        stt = self._ai.stt(wav_bytes, language=language)
        if stt.get("status") != "ok":
            self._set_state("idle")
            return {"status": stt.get("status", "failed"),
                    "error": stt.get("error"), "stage": "stt"}
        transcript = str(stt.get("text") or "").strip()
        if not transcript:
            self._set_state("idle")
            return {"status": "ok", "transcript": "", "reply": "",
                    "note": "no speech detected"}
        self._set_state("processing")
        chat = self._assistant.chat(transcript)
        if chat.get("status") != "ok":
            self._set_state("idle")
            return {"status": chat.get("status", "failed"), "transcript": transcript,
                    "error": chat.get("error"), "stage": "chat"}
        reply = str(chat.get("text") or "").strip()
        out: Dict[str, Any] = {"status": "ok", "transcript": transcript,
                               "reply": reply}
        if speak and reply:
            self._set_state("speaking")
            tts = self._ai.tts(reply)
            if tts.get("status") == "ok":
                out["audio_b64"] = tts.get("audio_b64")
                out["audio_format"] = tts.get("format", "wav")
            else:
                out["tts_error"] = tts.get("error")
        self._set_state("idle")
        self.turns += 1
        self._event("voice_turn", transcript=transcript, reply=reply[:200],
                    turns=self.turns)
        return out

    def close(self) -> None:
        self._set_state("idle")


class VoiceManager:
    """Owns the at-most-one live voice session + the gesture hook."""

    def __init__(self, ai: Any, assistant: Any,
                 feedback_for: Callable[[str], Optional[Callable[[str], Any]]],
                 emit_event: Optional[Callable[[str, Dict[str, Any]], Any]] = None):
        self._ai = ai
        self._assistant = assistant
        self._feedback_for = feedback_for
        self._emit = emit_event
        self._session: Optional[VoiceSession] = None

    def session(self) -> Optional[VoiceSession]:
        if self._session and self._session.expired():
            self._session.close()
            self._session = None
        return self._session

    def open(self, device_id: str = "") -> VoiceSession:
        cur = self.session()
        if cur is not None:
            cur.last_activity = time.time()
            return cur
        self._session = VoiceSession(
            self._ai, self._assistant,
            feedback=self._feedback_for(device_id),
            emit_event=self._emit, device_id=device_id)
        return self._session

    def close(self) -> None:
        if self._session:
            self._session.close()
            self._session = None

    def on_gesture(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """wearable.gesture.v1 value → maybe open/close a voice session.

        ``double_tap`` or the ``thoth_voice_toggle`` action toggles a session;
        ``thoth_voice_end``/``long_press`` always ends it. Other gestures are
        ignored (``{"handled": False}``).
        """
        gesture = str(event.get("gesture") or "")
        action = str(event.get("action") or "")
        device_id = str(event.get("subject") or "")
        if gesture == "double_tap" or action == "thoth_voice_toggle":
            cur = self.session()
            if cur is not None and cur.turns == 0:
                # Empty session re-toggled → treat as cancel.
                self.close()
                return {"handled": True, "session": "closed"}
            self.open(device_id)
            return {"handled": True, "session": "open"}
        if action in ("thoth_voice_end",) or gesture == "long_press":
            if self._session is not None:
                self.close()
                return {"handled": True, "session": "closed"}
            return {"handled": True, "session": "none"}
        return {"handled": False}


__all__ = ["VoiceSession", "VoiceManager"]
