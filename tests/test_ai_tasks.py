"""AIServices + TaskStore — local STT/TTS/LLM lifecycle, task folders."""
import json
import time

import pytest

from thoth.ai_services import AIServices
from thoth.tasks import TaskStore


@pytest.fixture
def ai(tmp_path):
    return AIServices(lambda: {"ai": {}}, tmp_path / "ai")


def test_status_degrades_gracefully(ai):
    st = ai.status()
    assert st["stt"]["status"] in ("installable", "not_installed")
    assert st["tts"]["status"] == "not_installed"
    assert st["llm"]["status"] == "stopped"


def test_tts_unavailable_without_piper(ai):
    out = ai.tts("hello")
    assert out["status"] == "unavailable"


def test_chat_unavailable_without_llama(ai):
    out = ai.chat([{"role": "user", "content": "hi"}])
    assert out["status"] == "unavailable"
    assert "not configured" in out["error"] or "llama" in out["error"]


def test_llm_config_from_config(tmp_path):
    ai = AIServices(lambda: {"ai": {"llm": {"port": 9999, "model": "m.gguf",
                                          "threads": 2}}},
                    tmp_path / "ai")
    assert ai._llm_port() == 9999
    assert ai._llm_url() == "http://127.0.0.1:9999"


# -- tasks ----------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    root = tmp_path / "tasks_root"
    return TaskStore(lambda: str(root), lambda: {"tasks": {}}, ai=None)


def test_task_folder_structure(store):
    m = store.create("Weekly steps chart", "Chart my step count by week")
    d = store.root() / m["id"]
    for sub in ("inputs", "outputs", "code", "runs"):
        assert (d / sub).is_dir()
    assert m["route"] == "auto"
    listed = store.list()
    assert listed and listed[0]["id"] == m["id"]
    assert store.get(m["id"])["title"] == "Weekly steps chart"


def test_task_inputs_and_file_read(store):
    m = store.create("t", "goal")
    store.put_input(m["id"], "cfg.yaml", b"a: 1\n")
    store.put_input(m["id"], "data.csv", b"x,y\n1,2\n")
    files = store.list_files(m["id"], "inputs")
    assert {f["path"] for f in files} == {"inputs/cfg.yaml", "inputs/data.csv"}
    assert store.read_file(m["id"], "inputs/cfg.yaml") == b"a: 1\n"
    # path traversal refused
    assert store.read_file(m["id"], "../task.json") is None
    assert store.read_file(m["id"], "task.json") is None


def test_task_run_local_route_fails_cleanly_without_llm(store):
    m = store.create("report", "Write a report", route="local")
    res = store.run(m["id"])
    assert res["accepted"] and res["route"] == "local"
    for _ in range(50):
        time.sleep(0.1)
        got = store.get(m["id"])
        if got["status"] != "running":
            break
    got = store.get(m["id"])
    assert got["status"] == "failed"            # no ai wired (ai=None)
    assert got["runs"] and got["runs"][0]["status"] == "failed"


def test_task_run_unknown_id(store):
    with pytest.raises(KeyError):
        store.run("nope")
    with pytest.raises(KeyError):
        store.put_input("nope", "f", b"x")


def test_route_resolution(store, monkeypatch):
    monkeypatch.setattr(store, "_codex_bin", lambda: "/usr/bin/codex")
    assert store._resolve_route("auto") == "codex"
    assert store._resolve_route("local") == "local"
    monkeypatch.setattr(store, "_codex_bin", lambda: None)
    assert store._resolve_route("auto") == "local"


def test_set_root_relocates_and_validates(store, tmp_path):
    # Mounted-drive relocation: creates + probes the new root.
    new_root = tmp_path / "drive" / "ThothTasks"
    out = store.set_root(str(new_root))
    assert out == str(new_root)
    assert new_root.is_dir()
    assert not (new_root / ".write-probe").exists()
    # Relative and empty roots refused.
    with pytest.raises(ValueError):
        store.set_root("relative/path")
    with pytest.raises(ValueError):
        store.set_root("")
