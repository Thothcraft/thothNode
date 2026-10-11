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


def _pcm_stats(capture_dir: Path,
               pcm_files: List[str]) -> Optional[Dict[str, Any]]:
    """Audio level stats over the capture's PCM blobs — per-chunk
    RMS in dBFS plus absolute int16 amplitude stats, matching the
    ``value``/``level_dbfs`` field shape mic descriptors emit live."""
    import math
    import struct

    chunk_vals: List[float] = []   # absolute int16 sample amplitudes
    levels: List[float] = []       # dBFS per 100 ms chunk
    seen = set()
    for name in pcm_files:
        if name in seen:
            continue
        seen.add(name)
        p = Path(capture_dir) / name
        if not p.exists():
            continue
        try:
            raw = p.read_bytes()
        except OSError:
            continue
        n = len(raw) - (len(raw) % 2)
        try:
            import numpy as np  # faster path when available
            arr = np.frombuffer(raw[:n], dtype="<i2").astype(np.float64)
            chunk_vals.extend(np.abs(arr).tolist())
            frame = 3200  # 100 ms @ 16 kHz s16le
            for i in range(0, len(arr) - frame + 1, frame):
                rms = float(np.sqrt(np.mean(arr[i:i + frame] ** 2)))
                levels.append(20.0 * math.log10(max(rms, 1.0) / 32768.0))
        except ImportError:
            ints = struct.unpack(f"<{n // 2}h", raw[:n])
            chunk_vals.extend(abs(v) for v in ints[::4])  # stride — cheap
            frame = 3200
            for i in range(0, len(ints) - frame + 1, frame):
                seg = ints[i:i + frame]
                rms = math.sqrt(sum(v * v for v in seg) / len(seg))
                levels.append(20.0 * math.log10(max(rms, 1.0) / 32768.0))
        break  # one sidecar file per sensor stream
    if not levels:
        return None

    def _stats(vals: List[float]) -> Dict[str, float]:
        s = sorted(vals)
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        return {"mean": round(m, 3), "std": round(math.sqrt(var), 3),
                "min": round(s[0], 3), "max": round(s[-1], 3),
                "n": len(vals)}

    return {"value": _stats(chunk_vals),
            "level_dbfs": _stats(levels)}


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
        pcm_files: List[str] = []
        with jf.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    samples.append(_sample_from_row(row, device_id))
                    p = row.get("payload") or {}
                    if (isinstance(p, dict)
                            and p.get("encoding") == "pcm_s16le"
                            and isinstance(p.get("file"), str)):
                        pcm_files.append(p["file"])
                except Exception:
                    continue
        if not samples:
            continue
        desc = sensor_descriptor(samples, detail="full")
        desc["n"] = len(samples)
        if pcm_files:
            audio = _pcm_stats(Path(capture_dir), pcm_files)
            if audio:
                # Same `value` field name live windows emit for the mic —
                # keeps baseline-vs-window comparisons symmetric.
                fields = desc.setdefault("fields", {})
                fields.setdefault("value", audio["value"])
                fields.setdefault("level_dbfs", audio["level_dbfs"])
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
