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
_RUN_TIMEOUT_S = 1800


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
