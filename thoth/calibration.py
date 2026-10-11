"""Labeled descriptor calibrations — measured per-sensor baselines.

A calibration answers "what does <label> look like to this node's
sensors?" — e.g. ``sleep``, ``empty_room``, ``away``. It stores the same
physical-descriptor stats the context builder consumes (per-field
mean/std/min/max, scan emitters, on-demand fields) so agents can compare
a live window against a measured baseline instead of a fixed threshold.

Docs are plain JSON under ``~/.thoth/calibration/<label>.json`` and are
uploaded to Brain so the same baselines reach the context/chat agents.

    {"format": "calibration/v1", "label": "sleep",
     "device_id": "…", "capture_id": "…",
     "started_at": …, "stopped_at": …, "duration_s": 1200.0,
     "sensors": {"radio-cacd": {"type": "radio_env", "n": 240,
                               "fields": {…}, "scan": {…}}}}
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .settings import config_dir

FORMAT = "calibration/v1"


def calibration_dir() -> Path:
    d = config_dir() / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(label: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(label or "").strip()).strip("-.")
    return s or "calibration"


def _path(label: str) -> Path:
    return calibration_dir() / f"{_slug(label)}.json"


def list_calibrations() -> List[Dict[str, Any]]:
    out = []
    for f in sorted(calibration_dir().glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except Exception:
            continue
        sensors = doc.get("sensors") or {}
        out.append({"label": doc.get("label") or f.stem,
                    "created_at": doc.get("created_at"),
                    "duration_s": doc.get("duration_s"),
                    "capture_id": doc.get("capture_id"),
                    "sensors": sorted(sensors)})
    return out


def get(label: str) -> Optional[Dict[str, Any]]:
    p = _path(label)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def delete(label: str) -> bool:
    p = _path(label)
    if not p.exists():
        return False
    p.unlink()
    return True


def save(doc: Dict[str, Any]) -> Dict[str, Any]:
    doc["format"] = FORMAT
    _path(str(doc.get("label") or "calibration")).write_text(
        json.dumps(doc, indent=2))
    return doc


def _sample_from_row(row: Dict[str, Any], device_id: str):
    from whispy.contracts import SensorSample
    return SensorSample(
        device_id=str(row.get("device_id") or device_id),
        sensor_id=str(row.get("sensor_id") or ""),
        sensor_type=str(row.get("sensor_type") or ""),
        timestamp=float(row.get("timestamp") or 0.0),
        sequence=int(row.get("sequence") or 0),
        payload_type=str(row.get("payload_type") or "auto"),
        payload=row.get("payload"),
        metadata=row.get("metadata") or {})


def build_from_capture(capture_dir: Path, manifest: Dict[str, Any],
                       label: str, *,
                       extra_sections: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
    """Per-sensor descriptor stats over one capture's JSONL files.

    Reuses whispy's ``sensor_descriptor`` at ``full`` detail — the
    calibration baseline is literally the same shape the context agent
    sees live, so comparisons are apples-to-apples."""
    from whispy.descriptors import sensor_descriptor

    device_id = str(manifest.get("device_id") or "")
    sensors: Dict[str, Any] = {}
    for jf in sorted(Path(capture_dir).glob("*.jsonl")):
        sid = jf.stem
        samples = []
        with jf.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(_sample_from_row(json.loads(line),
                                                    device_id))
                except Exception:
                    continue
        if not samples:
            continue
        desc = sensor_descriptor(samples, detail="full")
        desc["n"] = len(samples)
        sensors[sid] = desc
    for sid, sec in (extra_sections or {}).items():
        if isinstance(sec, dict):
            sensors.setdefault(sid, sec)

    started = float(manifest.get("started_at") or 0.0)
    stopped = float(manifest.get("stopped_at") or time.time())
    return {
        "format": FORMAT,
        "label": str(label),
        "device_id": device_id,
        "capture_id": manifest.get("id"),
        "created_at": time.time(),
        "started_at": started,
        "stopped_at": stopped,
        "duration_s": round(max(0.0, stopped - started), 1),
        "sensors": sensors,
    }
