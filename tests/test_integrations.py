"""DriveManager — rclone-backed Drive mount + scoped fs browse."""
import json

import pytest

from thoth.integrations import DriveManager
from thoth.settings import ConfigStore


@pytest.fixture
def cfg(tmp_home):
    cfg = ConfigStore()
    cfg.set("thoth_root", str(tmp_home))
    return cfg


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("THOTH_HOME", str(tmp_path))
    return tmp_path


def test_status_shape_without_rclone(tmp_home, cfg):
    d = DriveManager(cfg)
    s = d.status()
    assert set(s) >= {"installed", "configured", "mounted", "remote",
                      "mount", "install_hint"}
    assert s["configured"] is False
    assert s["mounted"] is False


def test_authorize_rejects_bad_token(tmp_home, cfg, monkeypatch):
    d = DriveManager(cfg)
    monkeypatch.setattr(DriveManager, "_rclone", staticmethod(lambda: "/bin/rclone"))
    out = d.authorize("not json")
    assert out["ok"] is False
    out = d.authorize(json.dumps({"access_token": "x"}))
    assert out["ok"] is False
    assert "token" in out["error"]


def test_authorize_creates_remote_and_mounts(tmp_home, cfg, monkeypatch):
    d = DriveManager(cfg)
    monkeypatch.setattr(DriveManager, "_rclone", staticmethod(lambda: "/bin/rclone"))
    calls = []

    def fake_run(self, *args, timeout=30.0):
        calls.append(args)
        return ""

    mounted = {}
    monkeypatch.setattr(DriveManager, "_run", fake_run)
    monkeypatch.setattr(DriveManager, "_configured",
                        lambda self: True)
    monkeypatch.setattr(DriveManager, "_mounted",
                        lambda self: mounted.setdefault("v", False) or mounted["v"])
    monkeypatch.setattr(__import__("subprocess"), "Popen",
                        lambda *a, **k: mounted.__setitem__("v", True) or object())
    out = d.authorize(json.dumps(
        {"access_token": "a", "refresh_token": "r",
         "token_type": "Bearer", "expiry": "2026-01-01T00:00:00Z"}))
    assert out["ok"] is True
    # config create landed on our remote with the pasted token
    assert calls and calls[0][:3] == ("config", "create", "gdrive")


def test_fs_browse_scoped_to_roots(tmp_home, cfg):
    d = DriveManager(cfg)
    home = d._home
    (home / "visible").mkdir()
    out = d.list_dir(str(home))
    assert out["ok"] is True
    names = {e["name"] for e in out["entries"]}
    assert "visible" in names
    # outside the allowlist → refused
    out = d.list_dir("/etc")
    assert out["ok"] is False
    # file, not dir → refused
    f = home / "a.txt"
    f.write_text("x")
    out = d.list_dir(str(f))
    assert out["ok"] is False
