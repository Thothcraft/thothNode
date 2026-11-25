"""Assistant memory/context, voice turns, task chat console, sensing
thresholds, and fused-IMU activity."""
import json
import time

import pytest

from thoth.assistant import Assistant
from thoth.voice import VoiceManager, VoiceSession
from thoth.tasks import TaskStore
from thoth.task_models import TaskModelHub


class FakeAI:
    """AIServices stand-in: canned chat/stt/tts results."""

    def __init__(self, reply="hi there", reflect_json=None):
        self.reply = reply
        self.reflect_json = reflect_json or {}
        self.calls = []
        self.stt_text = "what time is it"
        self.tts_audio = "QUJD"  # base64 "ABC"

    def chat(self, messages, max_tokens=512, temperature=0.7):
        self.calls.append(messages)
        user = messages[-1]["content"]
        if "Update the assistant's JSON memory" in user:
            return {"status": "ok", "text": json.dumps(self.reflect_json)}
        return {"status": "ok", "text": self.reply, "model": "fake"}

    def stt(self, wav_bytes, language=None):
        return {"status": "ok", "text": self.stt_text, "language": "en"}

    def tts(self, text):
        return {"status": "ok", "format": "wav", "audio_b64": self.tts_audio}


# -- assistant ------------------------------------------------------------------

def test_assistant_chat_updates_history(tmp_path):
    ai = FakeAI()
    a = Assistant(ai, tmp_path / "memory.json")
    out = a.chat("remember that my name is Gad")
    assert out["status"] == "ok" and out["text"] == "hi there"
    mem = a.memory()
    roles = [h["role"] for h in mem["history"]]
    assert roles == ["user", "assistant"]
    assert mem["history"][0]["text"] == "remember that my name is Gad"


def test_assistant_reflect_merges_fields(tmp_path):
    ai = FakeAI(reflect_json={
        "facts": {"name": "Gad"}, "people": {"Mom": "calls weekly"},
        "summary": "User is named Gad."})
    a = Assistant(ai, tmp_path / "memory.json")
    a.chat("my name is Gad")
    mem = a.memory()
    assert mem["facts"]["name"] == "Gad"
    assert mem["people"]["Mom"] == "calls weekly"
    assert mem["summary"] == "User is named Gad."
    # second call = reflect prompt
    assert len(ai.calls) == 2
    assert "JSON memory" in ai.calls[1][-1]["content"]


def test_assistant_context_timeline_in_prompt(tmp_path):
    ai = FakeAI()
    now = time.time()
    a = Assistant(ai, tmp_path / "memory.json",
                  context_fn=lambda s: [{"ts": now - 60, "kind": "prediction",
                                         "task": "zone", "label": "L_NEAR",
                                         "device": "room1"}])
    a.chat("what's happening", reflect=False)
    system = ai.calls[0][0]["content"]
    assert "L_NEAR" in system and "Recent events" in system


def test_assistant_memory_put_and_clear(tmp_path):
    ai = FakeAI()
    a = Assistant(ai, tmp_path / "memory.json")
    a.put_memory({"facts": {"coffee": "black"}, "settings": {"history_s": 60}})
    mem = a.memory()
    assert mem["facts"]["coffee"] == "black"
    assert mem["settings"]["history_s"] == 60
    a.put_memory({"facts": {"coffee": None}})          # null deletes
    assert "coffee" not in a.memory()["facts"]
    a.clear_memory()
    assert a.memory()["facts"] == {} and a.memory()["history"] == []


def test_assistant_degrades_without_llm(tmp_path):
    class DeadAI:
        def chat(self, m, max_tokens=512, temperature=0.7):
            return {"status": "unavailable", "error": "no model"}
    a = Assistant(DeadAI(), tmp_path / "memory.json")
    out = a.chat("hi")
    assert out["status"] == "unavailable"
    assert a.memory()["history"] == []                # failed turn not logged


# -- voice -----------------------------------------------------------------------

def test_voice_turn_stt_chat_tts(tmp_path):
    ai = FakeAI(reply="it is 3pm")
    a = Assistant(ai, tmp_path / "memory.json")
    states = []
    sess = VoiceSession(ai, a, feedback=states.append, device_id="buds-1")
    out = sess.turn(b"RIFF-fake-wav", speak=True)
    assert out["status"] == "ok"
    assert out["transcript"] == "what time is it"
    assert out["reply"] == "it is 3pm"
    assert out["audio_b64"] == "QUJD"
    # listening → processing → speaking → idle sequence pushed to buds
    assert "listening" in states and "speaking" in states
    assert states[-1] == "idle"
    assert sess.turns == 1


def test_voice_manager_gesture_toggles(tmp_path):
    ai = FakeAI()
    a = Assistant(ai, tmp_path / "memory.json")
    events = []
    vm = VoiceManager(ai, a, feedback_for=lambda did: lambda s: None,
                      emit_event=lambda k, d: events.append((k, d)))
    out = vm.on_gesture({"gesture": "double_tap", "subject": "buds-1"})
    assert out["handled"] and out["session"] == "open"
    assert vm.session() is not None
    # Second double-tap on an unused session cancels it
    out = vm.on_gesture({"gesture": "double_tap", "subject": "buds-1"})
    assert out["session"] == "closed"
    assert vm.session() is None
    # thoth_voice_end always ends
    vm.open("buds-1")
    out = vm.on_gesture({"action": "thoth_voice_end", "subject": "buds-1"})
    assert out["session"] == "closed"
    # unrelated gestures pass through
    assert not vm.on_gesture({"gesture": "single_tap"})["handled"]


def test_voice_session_expiry(tmp_path):
    ai = FakeAI()
    a = Assistant(ai, tmp_path / "memory.json")
    sess = VoiceSession(ai, a, device_id="buds-1")
    assert not sess.expired()
    sess.last_activity = time.time() - 999
    assert sess.expired()


# -- task chat console ------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return TaskStore(lambda: str(tmp_path / "tasks_root"),
                     lambda: {"tasks": {}}, ai=FakeAI())


def test_task_chat_local_writes_jsonl(store, tmp_path):
    m = store.create("report", "Summarize data", route="local")
    out = store.chat(m["id"], "write inputs/notes.md summarizing x")
    assert out["status"] == "ok"
    log = store.chat_log(m["id"])
    assert [e["role"] for e in log] == ["user", "assistant"]
    assert log[0]["text"].startswith("write inputs")
    # file listing returned with the turn
    assert "inputs" in out["files"]


def test_task_chat_extracts_fenced_files(store):
    m = store.create("files", "make files", route="local")
    store.ai = FakeAI(reply="done\n```inputs:notes.txt\nhello world\n```")
    out = store.chat(m["id"], "create a note")
    assert out["status"] == "ok"
    assert store.read_file(m["id"], "inputs/notes.txt") == b"hello world\n"


def test_task_chat_unknown(store):
    with pytest.raises(KeyError):
        store.chat("nope", "hi")


# -- sensing snapshot + thresholds ------------------------------------------------

def test_sensing_snapshot_zones(tmp_path):
    hub = TaskModelHub(tmp_path / "tm", config=lambda: {"sensing": {}})
    snap = hub.sensing_snapshot()
    assert set(snap["zones"]) == {"L_NEAR", "M_NEAR", "R_NEAR",
                                "L_FAR", "M_FAR", "R_FAR"}
    assert all(z["occupied"] is False for z in snap["zones"].values())
    assert snap["r_th_m"] > 0


def test_set_thresholds(tmp_path):
    hub = TaskModelHub(tmp_path / "tm", config=lambda: {"sensing": {}})
    snap = hub.set_thresholds(sectors={"L": 8.5, "M": 7.0},
                              zones={"L_NEAR": 9.0}, r_th_m=1.8)
    assert snap["sectors"]["L"]["threshold_db"] == 8.5
    assert snap["zones"]["L_NEAR"]["threshold_db"] == 9.0
    # un-set zone falls back to its sector threshold
    assert snap["zones"]["M_FAR"]["threshold_db"] == 7.0
    assert snap["r_th_m"] == 1.8
    # persisted + reloaded
    hub2 = TaskModelHub(tmp_path / "tm", config=lambda: {"sensing": {}})
    assert hub2.cal["thresholds"]["L"] == 8.5
    assert hub2.cal["zone_thresholds"]["L_NEAR"] == 9.0
    with pytest.raises(ValueError):
        hub.set_thresholds(zones={"NOPE": 1.0})
    with pytest.raises(ValueError):
        hub.set_thresholds(r_th_m=999.0)


# -- fused IMU activity ------------------------------------------------------------

def _imu(subject, amp=1.0):
    return {"schema": "imu.window.v1", "subject": subject,
            "source_id": subject, "timestamp": time.time(),
            "value": {"rate_hz": 50,
                      "axes": {"x": [0.0] * 64, "y": [0.0] * 64,
                               "z": [amp] * 64}}}


def test_fused_activity_two_subjects(tmp_path):
    hub = TaskModelHub(tmp_path / "tm", config=lambda: {"sensing": {}})
    p1 = hub.on_observation(_imu("watch-1"))
    assert p1["subject"] == "watch-1"              # single → per-subject
    p2 = hub.on_observation(_imu("necklace-1"))
    assert p2["subject"] == "fused"                # two fresh → fused
    assert set(p2["meta"]["sources"]) == {"watch-1", "necklace-1"}
    assert hub.latest["activity:fused"]["subject"] == "fused"
