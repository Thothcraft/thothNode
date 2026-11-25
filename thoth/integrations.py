"""Third-party integrations the node mounts into the local workspace.

Google Drive is the first: the user runs ``rclone authorize "drive"`` on
any machine with a browser, pastes the resulting token JSON into the app,
and the node writes an rclone remote, mounts it under ``~/GoogleDrive``,
and keeps it mounted across reboots (the daemon re-ensures on start).

``tasks.root`` can then point inside the mount so codex works directly on
files that sync back to Drive.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REMOTE = "gdrive"
_MOUNT_FLAG = "thoth-integrations"


def _default_mount(home: Path) -> Path:
    return home / "GoogleDrive"


class DriveManager:
    """rclone-backed Google Drive mount owned by the node."""

    def __init__(self, config) -> None:
        self._conf = config
        home = Path(os.path.expanduser(
            str(config.get("thoth_root") or "~")))
        if not home.is_dir():
            home = Path.home()
        self._home = home
        self._mount = Path(str(
            config.get("integrations.drive.mount") or _default_mount(home)))
        self._conf_dir = Path(str(
            config.get("integrations.drive.conf_dir")
            or home / ".config" / "rclone"))
        self._conf_file = self._conf_dir / "rclone.conf"

    # -- rclone plumbing ------------------------------------------------------

    @staticmethod
    def _rclone() -> Optional[str]:
        return shutil.which("rclone")

    def _env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env["RCLONE_CONFIG"] = str(self._conf_file)
        return env

    def _run(self, *args: str, timeout: float = 30.0) -> str:
        out = subprocess.run(
            [self._rclone() or "rclone", *args],
            env=self._env(), capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or out.stdout.strip()
                               or f"rclone exit {out.returncode}")
        return out.stdout.strip()

    def _configured(self) -> bool:
        if not self._conf_file.is_file():
            return False
        try:
            return f"[{REMOTE}]" in self._conf_file.read_text()
        except Exception:
            return False

    def _mounted(self) -> bool:
        if not self._mount.is_dir():
            return False
        try:
            # A live rclone mount answers `lsd` quickly; a stale mountpoint
            # (post-crash) raises — treat that as unmounted.
            self._run("lsd", f"{REMOTE}:", "--max-depth", "1", timeout=8.0)
            with open("/proc/mounts") as fh:
                if str(self._mount) in fh.read():
                    return True
        except Exception:
            pass
        # Fallback: mountpoint registered but rclone unreachable.
        try:
            with open("/proc/mounts") as fh:
                return str(self._mount) in fh.read()
        except Exception:
            return False

    # -- public surface ---------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        installed = self._rclone() is not None
        return {
            "installed": installed,
            "configured": self._configured(),
            "mounted": self._mounted() if installed and self._configured()
                       else False,
            "remote": REMOTE,
            "mount": str(self._mount),
            "tasks_root_on_drive": self._tasks_root_inside(),
            "install_hint": "curl https://rclone.org/install.sh | sudo bash",
        }

    def _tasks_root_inside(self) -> bool:
        try:
            from .settings import ConfigStore  # local import — avoid cycle
        except Exception:
            return False
        root = str(self._conf.get("tasks.root") or "")
        return bool(root) and root.startswith(str(self._mount))

    def authorize(self, token_json: str) -> Dict[str, Any]:
        """Write the rclone remote from a pasted ``rclone authorize`` token
        blob, then mount. The token carries the refresh_token so the mount
        renews itself indefinitely."""
        if self._rclone() is None:
            return {"ok": False, "error": "rclone not installed",
                    "install_hint": "curl https://rclone.org/install.sh | sudo bash"}
        try:
            parsed = json.loads(token_json)
            if "access_token" not in parsed or "refresh_token" not in parsed:
                return {"ok": False,
                        "error": "token looks wrong — paste the whole JSON blob "
                                 "rclone authorize prints, including the braces"}
        except ValueError:
            return {"ok": False, "error": "not valid JSON"}
        self._conf_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._run("config", "create", REMOTE, "drive",
                      "token", token_json, "scope", "drive",
                      "--non-interactive", timeout=30.0)
        except Exception as exc:
            return {"ok": False, "error": f"config create failed: {exc}"}
        return self.mount()

    def mount(self) -> Dict[str, Any]:
        if self._rclone() is None:
            return {"ok": False, "error": "rclone not installed"}
        if not self._configured():
            return {"ok": False, "error": "drive not linked yet"}
        if self._mounted():
            return {"ok": True, "already": True, **self._brief()}
        self._mount.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.Popen(
                [self._rclone() or "rclone", "mount", f"{REMOTE}:",
                 str(self._mount), "--vfs-cache-mode", "writes",
                 "--dir-cache-time", "30s",
                 "--volname", _MOUNT_FLAG],
                env=self._env(),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
        except Exception as exc:
            return {"ok": False, "error": f"mount failed: {exc}"}
        import time
        for _ in range(20):
            if self._mounted():
                self._conf.set("integrations.drive.mounted", True)
                return {"ok": True, **self._brief()}
            time.sleep(0.5)
        return {"ok": False,
                "error": "mount did not come up — check the Drive token"}

    def unmount(self) -> Dict[str, Any]:
        try:
            subprocess.run(["fusermount", "-u", str(self._mount)],
                           capture_output=True, timeout=15)
        except Exception:
            pass
        try:
            subprocess.run(["umount", "-l", str(self._mount)],
                           capture_output=True, timeout=15)
        except Exception:
            pass
        self._conf.set("integrations.drive.mounted", False)
        return {"ok": not self._mounted()}

    def ensure_mounted(self) -> None:
        """Daemon-start hook — re-mount after reboot if it was up before."""
        try:
            if (self._conf.get("integrations.drive.mounted")
                    and self._configured() and not self._mounted()):
                self.mount()
        except Exception as exc:
            logger.debug("drive ensure_mounted failed: %s", exc)

    def _brief(self) -> Dict[str, Any]:
        return {"mount": str(self._mount), "remote": REMOTE}

    # -- filesystem browse ------------------------------------------------------

    def browse_roots(self) -> List[Path]:
        roots = [self._home]
        for base in (Path("/media"), Path("/mnt")):
            try:
                for child in sorted(base.iterdir()):
                    if child.is_dir():
                        roots.append(child)
            except Exception:
                continue
        return roots

    def allowed(self, path: Path) -> bool:
        try:
            rp = path.resolve()
        except Exception:
            return False
        for root in self.browse_roots():
            try:
                if rp == root or root in rp.parents:
                    return True
            except Exception:
                continue
        return False

    def list_dir(self, path: str) -> Dict[str, Any]:
        p = Path(path).expanduser()
        if not self.allowed(p):
            return {"ok": False, "error": "path outside browsable roots"}
        if not p.is_dir():
            return {"ok": False, "error": "not a directory"}
        entries = []
        try:
            for child in sorted(p.iterdir()):
                try:
                    st = child.stat()
                    entries.append({
                        "name": child.name, "dir": child.is_dir(),
                        "size": st.st_size, "mtime": st.st_mtime})
                except OSError:
                    entries.append({"name": child.name, "dir": child.is_dir()})
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": str(p), "parent": str(p.parent),
                "entries": entries}


__all__ = ["DriveManager", "REMOTE"]
