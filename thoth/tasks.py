"""Task folders on the node (FR-TASK-*): goal → run → outputs.

Each task is a directory under the configured root (``tasks.root``,
default ``~/ThothTasks`` — the user points a mounted drive or bisync
target at it):

    <task_id>/
      task.json            manifest {id, title, goal, route, status, created_at}
      inputs/              user-selected files + editable configs
      code/                generated/runnable code and scripts
      outputs/             reports, figures, files produced by runs
      runs/<iso>.log       run log
      runs/<iso>.json      run record {route, status, started, ended, exit}

Execution routes:
  - ``codex`` — ``codex exec`` (config ``tasks.codex_bin`` or PATH) with
    the task dir as working directory.
  - ``local`` — one-shot local-LLM completion via ``AIServices.chat``;
    the reply is written to ``outputs/report.md``.

Runs execute on a background thread; ``runs/*.json`` records status so
clients can poll ``GET /api/v1/tasks/<id>``.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SUBDIRS = ("inputs", "outputs", "code", "runs")
_TASK_JSON = "task.json"
_CHAT_JSONL = "chat.jsonl"
_RUN_TIMEOUT_S = 1800
_CHAT_TIMEOUT_S = 600


def _safe_id(value: str) -> str:
    out = "".join(c for c in value if c.isalnum() or c in "-_")[:40]
    return out or uuid.uuid4().hex[:8]


class TaskStore:
    def __init__(self, root_getter: Callable[[], str],
                 config: Callable[[], Dict[str, Any]],
                 ai: Any = None):
        self._root_getter = root_getter
        self._config = config
        self.ai = ai
        self._lock = threading.RLock()
        self._runs: Dict[str, threading.Thread] = {}
        self._procs: Dict[str, subprocess.Popen] = {}

    # -- paths ---------------------------------------------------------------
    def root(self) -> Path:
        return Path(self._root_getter() or "~/ThothTasks").expanduser()

    def _dir(self, task_id: str) -> Path:
        return self.root() / _safe_id(task_id)

    def _read(self, task_id: str) -> Optional[Dict[str, Any]]:
        p = self._dir(task_id) / _TASK_JSON
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def _write(self, task_id: str, manifest: Dict[str, Any]) -> None:
        p = self._dir(task_id) / _TASK_JSON
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(p)

    # -- CRUD -------------------------------------------------------------------
    def create(self, title: str, goal: str,
               route: str = "auto") -> Dict[str, Any]:
        task_id = _safe_id(title.lower().replace(" ", "-")[:24]
                           + "-" + uuid.uuid4().hex[:6])
        d = self._dir(task_id)
        for sub in SUBDIRS:
            (d / sub).mkdir(parents=True, exist_ok=True)
        route = route if route in ("auto", "codex", "local") else "auto"
        manifest = {"id": task_id, "title": title, "goal": goal,
                    "route": route, "status": "created",
                    "created_at": time.time()}
        self._write(task_id, manifest)
        return manifest

    def list(self) -> List[Dict[str, Any]]:
        out = []
        if not self.root().exists():
            return out
        for child in sorted(self.root().iterdir()):
            m = self._read(child.name)
            if m:
                m["has_run"] = any((child / "runs").glob("*.json"))
                out.append(m)
        return out

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        m = self._read(task_id)
        if m is None:
            return None
        runs = []
        for p in sorted(self._dir(task_id).glob("runs/*.json")):
            try:
                runs.append(json.loads(p.read_text()))
            except Exception:
                pass
        m["runs"] = runs
        return m

    def put_input(self, task_id: str, name: str, data: bytes) -> Dict[str, Any]:
        if self._read(task_id) is None:
            raise KeyError(task_id)
        name = Path(name).name or "input.bin"
        path = self._dir(task_id) / "inputs" / name
        path.write_bytes(data)
        return {"path": f"inputs/{name}", "bytes": len(data)}

    def list_files(self, task_id: str, sub: str) -> List[Dict[str, Any]]:
        if sub not in SUBDIRS or self._read(task_id) is None:
            raise KeyError(task_id)
        base = self._dir(task_id) / sub
        return [{"path": p.relative_to(self._dir(task_id)).as_posix(),
                 "bytes": p.stat().st_size}
                for p in sorted(base.rglob("*")) if p.is_file()]

    def read_file(self, task_id: str, rel: str) -> Optional[bytes]:
        base = self._dir(task_id).resolve()
        target = (base / rel).resolve()
        if not str(target).startswith(str(base)) or not target.is_file():
            return None
        if str(target.relative_to(base).parts[0]) not in SUBDIRS:
            return None
        return target.read_bytes()

    # -- execution ----------------------------------------------------------------
    def _resolve_route(self, route: str) -> str:
        if route == "auto":
            return "codex" if self._codex_bin() else "local"
        return route

    def _codex_bin(self) -> Optional[str]:
        cfg = ((self._config() or {}).get("tasks") or {})
        exe = cfg.get("codex_bin") or shutil.which("codex")
        return str(exe) if exe else None

    def run(self, task_id: str) -> Dict[str, Any]:
        m = self._read(task_id)
        if m is None:
            raise KeyError(task_id)
        if task_id in self._runs and self._runs[task_id].is_alive():
            return {"accepted": False, "error": "task already running"}
        route = self._resolve_route(m.get("route") or "auto")
        stamp = time.strftime("%Y%m%dT%H%M%S")
        rec = {"run_id": stamp, "route": route, "status": "running",
               "started": time.time()}
        (self._dir(task_id) / "runs" / f"{stamp}.json").write_text(
            json.dumps(rec, indent=2))
        m["status"] = "running"
        m["route"] = m.get("route") if m.get("route") != "auto" else route
        self._write(task_id, m)
        t = threading.Thread(target=self._run_thread,
                             args=(task_id, m, rec, stamp), daemon=True)
        self._runs[task_id] = t
        t.start()
        return {"accepted": True, "run_id": stamp, "route": route}

    def cancel(self, task_id: str) -> Dict[str, Any]:
        proc = self._procs.get(task_id)
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
            return {"accepted": True}
        return {"accepted": False, "error": "no active run"}

    def _finish(self, task_id: str, stamp: str, rec: Dict[str, Any]) -> None:
        rec["ended"] = time.time()
        (self._dir(task_id) / "runs" / f"{stamp}.json").write_text(
            json.dumps(rec, indent=2))
        m = self._read(task_id)
        if m:
            m["status"] = rec["status"]
            self._write(task_id, m)

    def _run_thread(self, task_id: str, m: Dict[str, Any],
                    rec: Dict[str, Any], stamp: str) -> None:
        log_path = self._dir(task_id) / "runs" / f"{stamp}.log"
        try:
            if rec["route"] == "codex":
                self._run_codex(task_id, m, rec, log_path)
            else:
                self._run_local(task_id, m, rec, log_path)
        except Exception as exc:
            logger.warning("task %s run failed: %s", task_id, exc)
            rec["status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"
        self._procs.pop(task_id, None)
        self._finish(task_id, stamp, rec)

    def _run_codex(self, task_id: str, m: Dict[str, Any],
                   rec: Dict[str, Any], log_path: Path) -> None:
        exe = self._codex_bin()
        if not exe:
            rec["status"] = "failed"
            rec["error"] = "codex not installed on Room1"
            return
        goal = m.get("goal") or m.get("title") or ""
        inputs = [p.name for p in (self._dir(task_id) / "inputs").iterdir()]
        prompt = (f"You are running inside a task folder on Room1.\n"
                  f"Goal: {goal}\n"
                  f"inputs/ contains: {inputs or 'nothing'}\n"
                  f"Write all deliverables to outputs/. Put code in code/.\n")
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(
                [exe, "exec", "--full-auto", prompt],
                cwd=str(self._dir(task_id)),
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            self._procs[task_id] = proc
            try:
                rec["exit_code"] = proc.wait(timeout=_RUN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                rec["status"] = "failed"
                rec["error"] = "timeout"
                return
        rec["status"] = "succeeded" if rec.get("exit_code") == 0 else "failed"

    def _run_local(self, task_id: str, m: Dict[str, Any],
                   rec: Dict[str, Any], log_path: Path) -> None:
        if self.ai is None:
            rec["status"] = "failed"
            rec["error"] = "no local AI services"
            return
        goal = m.get("goal") or m.get("title") or ""
        result = self.ai.chat(
            [{"role": "system", "content":
              "You are a task-running assistant on Room1. Produce the "
              "requested deliverable as a Markdown report."},
             {"role": "user", "content": goal}],
            max_tokens=2048)
        log_path.write_text(json.dumps(result, indent=2))
        if result.get("status") != "ok":
            rec["status"] = "failed"
            rec["error"] = result.get("error")
            return
        out = self._dir(task_id) / "outputs" / "report.md"
        out.write_text(f"# {m.get('title')}\n\n{result.get('text', '')}\n")
        rec["status"] = "succeeded"
        rec["output"] = "outputs/report.md"

    # -- chat console (shell-like codex interaction) -----------------------------
    def _chat_path(self, task_id: str) -> Path:
        return self._dir(task_id) / _CHAT_JSONL

    def chat_log(self, task_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        if self._read(task_id) is None:
            raise KeyError(task_id)
        entries = []
        try:
            for line in self._chat_path(task_id).read_text().splitlines():
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.debug("chat log unreadable for %s: %s", task_id, exc)
        return entries[-limit:]

    def _chat_append(self, task_id: str, role: str, text: str,
                     **extra) -> Dict[str, Any]:
        entry = {"role": role, "text": text, "ts": time.time(), **extra}
        with self._lock:
            with self._chat_path(task_id).open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def _tree_summary(self, task_id: str) -> str:
        files = []
        base = self._dir(task_id)
        for sub in SUBDIRS:
            for p in sorted((base / sub).rglob("*")):
                if p.is_file():
                    files.append(p.relative_to(base).as_posix())
        return ", ".join(files[:80]) or "(empty task folder)"

    def chat(self, task_id: str, text: str,
             route: str = "auto") -> Dict[str, Any]:
        """Shell-like turn inside the task folder — the user message is
        recorded, then codex (or the local LLM) acts in the task dir and
        its reply is recorded. Files created by codex appear under
        code/outputs and are listed back to the client."""
        m = self._read(task_id)
        if m is None:
            raise KeyError(task_id)
        text = (text or "").strip()
        if not text:
            return {"status": "failed", "error": "empty message"}
        self._chat_append(task_id, "user", text)
        route = self._resolve_route(route or m.get("route") or "auto")
        if route == "codex":
            out = self._chat_codex(task_id, m, text)
        else:
            out = self._chat_local(task_id, m, text)
        entry = self._chat_append(
            task_id, "assistant", out.get("text") or out.get("error") or "",
            route=route, status=out.get("status"))
        out["entry"] = entry
        out["files"] = {"inputs": self.list_files(task_id, "inputs"),
                        "code": self.list_files(task_id, "code"),
                        "outputs": self.list_files(task_id, "outputs")}
        return out

    def _chat_codex(self, task_id: str, m: Dict[str, Any],
                    text: str) -> Dict[str, Any]:
        exe = self._codex_bin()
        if not exe:
            return {"status": "unavailable", "error": "codex not installed on Room1"}
        goal = m.get("goal") or m.get("title") or ""
        recent = self.chat_log(task_id, limit=12)
        history = "\n".join(f"{e['role'].upper()}: {e['text']}" for e in recent[:-1])
        prompt = (
            f"You are running inside the task folder {task_id} on Room1.\n"
            f"Goal: {goal}\n"
            f"Task folder now contains: {self._tree_summary(task_id)}\n"
            + (f"Recent session:\n{history}\n" if history else "")
            + f"USER: {text}\n\n"
            "Do the requested work — create inputs you need, write code to "
            "code/, deliverables to outputs/, then briefly report what you "
            "did and any file paths.")
        try:
            proc = subprocess.run(
                [exe, "exec", "--full-auto", prompt],
                cwd=str(self._dir(task_id)),
                capture_output=True, timeout=_CHAT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return {"status": "failed", "error": f"codex timed out after {_CHAT_TIMEOUT_S}s"}
        except Exception as exc:
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        text_out = proc.stdout.decode(errors="replace").strip()
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="replace").strip() or text_out
            return {"status": "failed", "error": err[:500] or f"codex exit {proc.returncode}"}
        # Last non-empty line block is the assistant-facing summary.
        tail = "\n".join(text_out.splitlines()[-30:])[:4000]
        return {"status": "ok", "text": tail or "(codex finished with no output)"}

    def _chat_local(self, task_id: str, m: Dict[str, Any],
                    text: str) -> Dict[str, Any]:
        if self.ai is None:
            return {"status": "unavailable", "error": "no local AI services"}
        goal = m.get("goal") or m.get("title") or ""
        recent = self.chat_log(task_id, limit=12)
        messages = [{"role": "system", "content": (
            "You are the task shell for a project folder on Room1. The "
            "folder has inputs/, code/, outputs/, runs/. Answer briefly "
            "and concretely; when asked to produce a file, output its "
            "content in a fenced block so it can be saved.")},
            {"role": "user", "content":
             f"Task: {goal}\nFolder: {self._tree_summary(task_id)}"}]
        messages += [{"role": e["role"], "content": e["text"]}
                     for e in recent if e.get("role") in ("user", "assistant")]
        out = self.ai.chat(messages, max_tokens=1024)
        if out.get("status") != "ok":
            return out
        text_out = str(out.get("text") or "")
        self._extract_files(task_id, text_out)
        return {"status": "ok", "text": text_out}

    def _extract_files(self, task_id: str, text: str) -> int:
        """Local-LLM replies can carry ```path:name``` fenced blocks —
        write them into the task folder (shell-like file creation)."""
        import re
        written = 0
        for match in re.finditer(
                r"```(\w+):([A-Za-z0-9_./-]+)\n(.*?)```", text, re.DOTALL):
            sub, name, body = match.group(1), match.group(2), match.group(3)
            if sub not in SUBDIRS or ".." in name:
                continue
            path = self._dir(task_id) / sub / Path(name).name
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((body.rstrip() + "\n").encode("utf-8"))
                written += 1
            except OSError:
                pass
        return written
