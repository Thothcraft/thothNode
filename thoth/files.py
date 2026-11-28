"""Sandboxed file roots for the app's browse-all Files page.

Named roots — ``shared`` (a drop zone under the node home), ``tasks``
(the task store's working root so codex jobs can consume uploads),
``captures`` (ring-buffered capture manifests), and ``drive`` (the
rclone Google Drive mount when linked). Every operation resolves the
requested path and refuses escapes outside the owning root.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


def _home(conf) -> Path:
    home = Path(os.path.expanduser(str(conf.get("thoth_root") or "~")))
    return home if home.is_dir() else Path.home()


class FileHub:
    def __init__(self, daemon) -> None:
        self._d = daemon

    # -- roots ----------------------------------------------------------------

    def roots(self) -> List[Dict[str, Any]]:
        d = self._d
        shared = _home(d.config) / "shared"
        shared.mkdir(parents=True, exist_ok=True)
        out: List[Dict[str, Any]] = [
            {"id": "shared", "name": "Shared", "path": str(shared)},
            {"id": "tasks", "name": "Task files",
             "path": str(d.tasks.root())},
            {"id": "captures", "name": "Captures",
             "path": str(d.captures.root), "read_only": True},
        ]
        status = d.drive.status()
        if status.get("installed"):
            out.append({"id": "drive", "name": "Google Drive",
                        "path": status.get("mount"),
                        "mounted": bool(status.get("mounted"))})
        return out

    def _root(self, root_id: str) -> Path:
        for r in self.roots():
            if r["id"] == root_id:
                p = Path(str(r["path"]))
                if not p.is_dir():
                    raise ValueError(f"root {root_id!r} not available")
                return p.resolve()
        raise ValueError(f"unknown root {root_id!r}")

    def _resolve(self, root_id: str, rel: str, must_exist: bool = False) -> Path:
        base = self._root(root_id)
        target = (base / rel.lstrip("/")).resolve()
        if target != base and base not in target.parents:
            raise ValueError("path escapes root")
        if must_exist and not target.exists():
            raise FileNotFoundError(rel)
        return target

    # -- operations -------------------------------------------------------------

    def list(self, root_id: str, rel: str) -> Dict[str, Any]:
        target = self._resolve(root_id, rel, must_exist=True)
        if not target.is_dir():
            raise ValueError("not a directory")
        entries = []
        for e in sorted(target.iterdir(),
                        key=lambda p: (not p.is_dir(), p.name.lower())):
            try:
                st = e.stat()
            except OSError:
                continue
            entries.append({"name": e.name, "dir": e.is_dir(),
                            "size": st.st_size,
                            "mtime": int(st.st_mtime)})
        return {"root": root_id, "path": rel, "entries": entries}

    def read(self, root_id: str, rel: str) -> Path:
        target = self._resolve(root_id, rel, must_exist=True)
        if not target.is_file():
            raise ValueError("not a file")
        return target

    def write(self, root_id: str, rel: str, data: bytes) -> Dict[str, Any]:
        target = self._resolve(root_id, rel)
        if target.is_dir():
            raise ValueError("path is a directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        return {"root": root_id, "path": rel, "size": len(data)}

    def mkdir(self, root_id: str, rel: str) -> Dict[str, Any]:
        target = self._resolve(root_id, rel)
        target.mkdir(parents=True, exist_ok=True)
        return {"root": root_id, "path": rel}

    def delete(self, root_id: str, rel: str, recursive: bool = False) -> Dict[str, Any]:
        target = self._resolve(root_id, rel, must_exist=True)
        base = self._root(root_id)
        if target == base:
            raise ValueError("cannot delete the root")
        if target.is_dir():
            if recursive:
                shutil.rmtree(target)
            else:
                target.rmdir()
        else:
            target.unlink()
        return {"root": root_id, "path": rel, "deleted": True}

    def copy_within(self, root_id: str, src_rel: str, dst_rel: str) -> Dict[str, Any]:
        src = self._resolve(root_id, src_rel, must_exist=True)
        dst = self._resolve(root_id, dst_rel)
        if dst.is_dir():
            dst = dst / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return {"root": root_id, "from": src_rel, "to": str(dst.relative_to(self._root(root_id)))}


def describe_error(exc: Exception) -> tuple[int, Dict[str, Any]]:
    if isinstance(exc, FileNotFoundError):
        return 404, {"error": "not found"}
    if isinstance(exc, (ValueError, PermissionError)):
        return 400, {"error": str(exc)}
    return 500, {"error": type(exc).__name__}
