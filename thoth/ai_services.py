"""Local AI services on the node (D-12): smallest practical models.

- **STT** — faster-whisper (CTranslate2 int8), variant from
  ``ai.stt.model`` (default ``base.en``). Runs in-process under a
  memory-gated lock; refuses to load under memory pressure.
- **TTS** — piper binary (``ai.tts.bin`` + ``ai.tts.voice``), wav out.
- **LLM** — llama.cpp ``llama-server`` managed subprocess
  (``ai.llm.bin`` + ``ai.llm.model``, OpenAI-compatible endpoint on
  ``ai.llm.port``, default 127.0.0.1:8081). Auto-starts on first chat
  and unloads after ``ai.llm.idle_s`` (default 30 min) — the 8 GB
  budget cannot hold everything at once.

Every service degrades to ``{"status": "not_installed", ...}`` when its
binary/model is absent, so the API surface is stable before the Room1
image ships the binaries.
"""

from __future__ import annotations

import base64
import json
import logging
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_LLM_IDLE_S = 1800.0
_LLM_START_TIMEOUT_S = 60.0


def _free_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        return None


class AIServices:
    """Owns STT/TTS/LLM lifecycles; one instance per daemon."""

    def __init__(self, config: Callable[[], Dict[str, Any]],
                 state_dir: Path):
        self._config = config
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._stt = None
        self._stt_err: Optional[str] = None
        self._llm_proc: Optional[subprocess.Popen] = None
        self._llm_last_used = 0.0
        self._lock = threading.RLock()          # serializes heavy loads
        self._reaper: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _cfg(self) -> Dict[str, Any]:
        cfg = (self._config() or {}).get("ai") or {}
        return cfg if isinstance(cfg, dict) else {}

    # -- STT ----------------------------------------------------------------
    def _load_stt(self) -> bool:
        if self._stt is not None:
            return True
        cfg = self._cfg().get("stt") or {}
        model = str(cfg.get("model") or "base.en")
        free = _free_mb()
        if free is not None and free < 768.0:
            self._stt_err = f"insufficient memory ({free:.0f} MiB free)"
            return False
        try:
            from faster_whisper import WhisperModel
            self._stt = WhisperModel(
                model, device="cpu",
                compute_type=str(cfg.get("compute_type") or "int8"),
                download_root=str(cfg.get("cache") or
                                  self.state_dir / "stt_models"))
            self._stt_err = None
            return True
        except Exception as exc:
            self._stt_err = f"{type(exc).__name__}: {exc}"
            return False

    def stt(self, wav_bytes: bytes,
            language: Optional[str] = None) -> Dict[str, Any]:
        """WAV bytes → transcript (offline, local only)."""
        with self._lock:
            if not self._load_stt():
                return {"status": "unavailable", "error": self._stt_err,
                        "hint": "install faster-whisper on the node"}
            tmp = self.state_dir / f"stt_{int(time.time() * 1000)}.wav"
            try:
                tmp.write_bytes(wav_bytes)
                cfg = self._cfg().get("stt") or {}
                segments, info = self._stt.transcribe(
                    str(tmp), language=language or cfg.get("language"),
                    vad_filter=True)
                text = " ".join(s.text.strip() for s in segments).strip()
                return {"status": "ok", "text": text,
                        "language": getattr(info, "language", None),
                        "model": self._cfg().get("stt", {}).get("model", "base.en")}
            except Exception as exc:
                return {"status": "failed",
                        "error": f"{type(exc).__name__}: {exc}"}
            finally:
                tmp.unlink(missing_ok=True)

    # -- TTS ----------------------------------------------------------------
    def _tts_paths(self):
        cfg = self._cfg().get("tts") or {}
        exe = cfg.get("bin") or shutil.which("piper")
        voice = cfg.get("voice")
        if exe:
            exe = str(exe)
        if voice:
            voice = str(voice)
        return exe, voice

    def tts(self, text: str) -> Dict[str, Any]:
        """text → WAV bytes (base64) via piper."""
        exe, voice = self._tts_paths()
        if not exe or not Path(exe).exists():
            return {"status": "unavailable", "error": "piper not installed",
                    "hint": "set ai.tts.bin + ai.tts.voice"}
        if not voice or not Path(voice).exists():
            return {"status": "unavailable", "error": "no piper voice configured"}
        with self._lock:
            try:
                proc = subprocess.run(
                    [exe, "--model", voice, "--output_file", "-"],
                    input=text.encode("utf-8"), capture_output=True,
                    timeout=120)
                if proc.returncode != 0 or not proc.stdout:
                    return {"status": "failed",
                            "error": proc.stderr.decode(errors="replace")[:300]}
                return {"status": "ok", "format": "wav",
                        "audio_b64": base64.b64encode(proc.stdout).decode()}
            except Exception as exc:
                return {"status": "failed",
                        "error": f"{type(exc).__name__}: {exc}"}

    # -- LLM ----------------------------------------------------------------
    def _llm_cfg(self) -> Dict[str, Any]:
        return self._cfg().get("llm") or {}

    def _llm_port(self) -> int:
        return int(self._llm_cfg().get("port") or 8081)

    def _llm_url(self) -> str:
        return f"http://127.0.0.1:{self._llm_port()}"

    def _llm_up(self) -> bool:
        if self._llm_proc is not None and self._llm_proc.poll() is None:
            return True
        try:
            urllib.request.urlopen(self._llm_url() + "/health", timeout=1)
            return True
        except Exception:
            return False

    def _start_llm(self) -> Optional[str]:
        cfg = self._llm_cfg()
        exe = cfg.get("bin") or shutil.which("llama-server")
        model = cfg.get("model")
        if not exe or not model:
            return "llama-server or model not configured"
        free = _free_mb()
        if free is not None and free < 2200.0:
            return f"insufficient memory ({free:.0f} MiB free)"
        args = [str(exe), "-m", str(model),
                "--port", str(self._llm_port()),
                "-t", str(int(cfg.get("threads") or 3)),
                "-c", str(int(cfg.get("ctx") or 4096)),
                "--no-webui", "--log-disable"]
        try:
            self._llm_proc = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        deadline = time.time() + _LLM_START_TIMEOUT_S
        while time.time() < deadline:
            if self._llm_proc.poll() is not None:
                self._llm_proc = None
                return "llama-server exited during startup"
            try:
                urllib.request.urlopen(self._llm_url() + "/health", timeout=1)
                self._llm_last_used = time.time()
                self._ensure_reaper()
                return None
            except Exception:
                time.sleep(0.5)
        return "llama-server start timed out"

    def _ensure_reaper(self) -> None:
        if self._reaper and self._reaper.is_alive():
            return
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True)
        self._reaper.start()

    def _reap_loop(self) -> None:
        idle = float(self._llm_cfg().get("idle_s") or _LLM_IDLE_S)
        while not self._stop.wait(30.0):
            if (self._llm_proc is not None and self._llm_last_used
                    and time.time() - self._llm_last_used > idle):
                logger.info("llm idle unload (%.0fs)", idle)
                self.stop_llm()
                return

    def stop_llm(self) -> None:
        proc, self._llm_proc = self._llm_proc, None
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def chat(self, messages: List[Dict[str, str]],
             max_tokens: int = 512,
             temperature: float = 0.7) -> Dict[str, Any]:
        """OpenAI-compatible chat via local llama-server."""
        with self._lock:
            if not self._llm_up():
                err = self._start_llm()
                if err:
                    return {"status": "unavailable", "error": err,
                            "hint": "set ai.llm.bin + ai.llm.model"}
        try:
            req = urllib.request.Request(
                self._llm_url() + "/v1/chat/completions",
                data=json.dumps({"messages": messages,
                                 "max_tokens": max_tokens,
                                 "temperature": temperature}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                out = json.loads(resp.read() or b"{}")
            self._llm_last_used = time.time()
            choice = (out.get("choices") or [{}])[0]
            return {"status": "ok",
                    "text": (choice.get("message") or {}).get("content", ""),
                    "model": self._llm_cfg().get("model"),
                    "usage": out.get("usage")}
        except Exception as exc:
            return {"status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"}

    # -- status / lifecycle ------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        exe, voice = self._tts_paths()
        stt_state = ("loaded" if self._stt is not None else
                     "installable" if self._stt_backend_available() else
                     "not_installed")
        return {
            "stt": {"status": stt_state,
                    "model": (self._cfg().get("stt") or {}).get("model", "base.en"),
                    "error": self._stt_err},
            "tts": {"status": "ok" if (exe and voice and Path(exe).exists()
                                       and Path(voice).exists())
                    else "not_installed",
                    "bin": exe, "voice": voice},
            "llm": {"status": "running" if self._llm_up() else "stopped",
                    "bin": self._llm_cfg().get("bin"),
                    "model": self._llm_cfg().get("model"),
                    "port": self._llm_port(),
                    "idle_s": self._llm_cfg().get("idle_s") or _LLM_IDLE_S},
        }

    @staticmethod
    def _stt_backend_available() -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def close(self) -> None:
        self._stop.set()
        self.stop_llm()
