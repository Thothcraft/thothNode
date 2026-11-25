"""Two models per sensing task, switchable at runtime (thothcraft-omi D-10).

Tasks and their registered models::

    presence  presence-snr@1 (light)      per-direction radar SNR threshold
              presence-occ-hier@2 (heavy) radar-repo occ-rf-hier late fusion
    zone      zone-snr@1 (light)          strongest SNR sector + range threshold
              zone-clf@1 (heavy)          classifier trained on room calibration
    activity  har-rules@1 (light)         IMU variance/frequency/inclination rules
              har-gbm@1 (heavy)           gradient boosting on activity calibration

The selection lives in ``models_active.json``; calibration artefacts
(empty-room background, per-sector thresholds, ``r_th``, trained
classifiers) in ``task_models/``. All processing stays on the node.

Inputs:

* radar — ``radar_frame`` samples carrying ``views_b64`` (rd/ra/re/xy
  24×24 log1p, ``whispy_sensor_mmwhat.dsp``); the RA view's rows are range
  bins, columns azimuth −60…+60°.
* CSI — ``csi_raw`` samples (``iq`` int8 pairs) from the left/right ESP32
  links; side is taken from config ``sensing.csi_sides`` or the sample's
  ``side`` field, otherwise sorted sensor ids → L, R.
* IMU — ``imu.window.v1`` observations (``axes`` x/y/z in g, ``rate_hz``).
"""

from __future__ import annotations

import base64
import json
import logging
import math
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TASKS: Dict[str, Tuple[str, str]] = {
    "presence": ("presence-snr@1", "presence-occ-hier@2", "presence-rf-radar@3"),
    "zone": ("zone-snr@1", "zone-clf@1"),
    "activity": ("har-rules@1", "har-gbm@1"),
}
ZONES = ("L_NEAR", "M_NEAR", "R_NEAR", "L_FAR", "M_FAR", "R_FAR")
SECTORS = ("L", "M", "R")
ACTIVITY_STEPS: List[Dict[str, Any]] = [
    {"label": "lying", "activity": "still", "posture": "lying", "duration_s": 60},
    {"label": "sitting", "activity": "still", "posture": "sitting", "duration_s": 60},
    {"label": "standing", "activity": "still", "posture": "standing", "duration_s": 60},
    {"label": "walking", "activity": "walking", "posture": "standing", "duration_s": 90},
    {"label": "running", "activity": "running", "posture": "standing", "duration_s": 45, "optional": True},
    {"label": "stairs", "activity": "stairs", "posture": "standing", "duration_s": 45, "optional": True},
]
ROOM_STEPS: List[Dict[str, Any]] = (
    [{"label": "EMPTY", "duration_s": 60}]
    + [{"label": z, "duration_s": 30} for z in ZONES]
)

_ANGLES = np.linspace(-60.0, 60.0, 24)
_EPS = 1e-6


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def decode_views(payload: Any) -> Optional[np.ndarray]:
    """radar payload → (4, 24, 24) float32 log1p views, or None."""
    if not isinstance(payload, dict):
        return None
    raw = payload.get("views_b64")
    if not raw:
        return None
    try:
        v = np.frombuffer(base64.b64decode(raw), dtype=np.float16)
        return v.reshape(4, 24, 24).astype(np.float32)
    except Exception:
        return None


RADAR_VIEW_NAMES = ("rd", "ra", "re", "xy")


def radar_window_features(frames: List[Tuple[np.ndarray, float]]) -> Optional[Dict[str, float]]:
    """5 s radar window → the frozen radar-repo feature row.

    Mirrors ``radar/E1/extract_multilink.extract_radar`` (per-window sum,
    sum-of-squares, abs frame-diff) and ``E1/features._radar_window_feats``
    (per-view mean / p90 / std p90 / mean delta / peak + SNR mean/max), the
    inputs ``E2/outputs/frozen/RF__radar.joblib`` was trained on.
    """
    if len(frames) < 5:
        return None
    views = np.stack([f[0] for f in frames]).astype(np.float64)   # (n, 4, 24, 24)
    n = views.shape[0]
    mean_img = views.mean(axis=0)
    std_img = np.sqrt(np.clip((views ** 2).mean(axis=0) - mean_img ** 2, 0, None))
    delta = np.abs(np.diff(views, axis=0)).sum(axis=0) / max(n - 1, 1)
    snr = np.asarray([f[1] for f in frames], dtype=np.float64)
    out: Dict[str, float] = {
        "snr_mean": float(np.nanmean(snr)) if np.isfinite(snr).any() else float("nan"),
        "snr_max": float(np.nanmax(snr)) if np.isfinite(snr).any() else float("nan"),
    }
    for vi, v in enumerate(RADAR_VIEW_NAMES):
        mi, si, di = mean_img[vi], std_img[vi], delta[vi]
        out[f"{v}_mean"] = float(mi.mean())
        out[f"{v}_p90"] = float(np.quantile(mi, 0.9))
        out[f"{v}_std90"] = float(np.quantile(si, 0.9))
        out[f"{v}_delta"] = float(di.mean())
        out[f"{v}_peak"] = float(mi.max())
    return out


def csi_amplitude(payload: Any) -> Optional[np.ndarray]:
    """csi_raw payload → per-subcarrier amplitude."""
    if not isinstance(payload, dict):
        return None
    iq = payload.get("iq")
    if not iq or len(iq) < 4:
        return None
    arr = np.asarray(iq, dtype=np.float32)
    arr = arr[: (len(arr) // 2) * 2].reshape(-1, 2)
    return np.sqrt((arr ** 2).sum(axis=1))


def link_energy(amps: List[np.ndarray]) -> float:
    """Temporal coefficient of variation of the mean amplitude (motion ↑)."""
    if len(amps) < 4:
        return float("nan")
    n = min(len(a) for a in amps)
    m = np.array([a[:n].mean() for a in amps], dtype=np.float64)
    return float(m.std() / (abs(m.mean()) + _EPS))


def imu_features(value: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """imu.window.v1 value → window features (g units)."""
    axes = (value or {}).get("axes") or {}
    xs, ys, zs = (np.asarray(axes.get(k) if axes.get(k) is not None else [],
                            dtype=np.float64) for k in "xyz")
    n = min(len(xs), len(ys), len(zs))
    if n < 8:
        return None
    xs, ys, zs = xs[:n], ys[:n], zs[:n]
    mag = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    if mag.mean() > 4.0:
        # Source is in m/s² (e.g. Android IMU), not g — normalize so the
        # g-unit thresholds below apply to both conventions.
        xs, ys, zs = xs / 9.80665, ys / 9.80665, zs / 9.80665
        mag = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    rate = float(value.get("rate_hz") or 10.0)
    dyn = mag - mag.mean()
    spec = np.abs(np.fft.rfft(dyn)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / max(rate, 1e-3))
    total = float(spec[1:].sum()) + _EPS
    band = float(spec[(freqs >= 1.0) & (freqs <= 3.0)].sum())
    dom = float(freqs[1:][int(np.argmax(spec[1:]))]) if n > 2 else 0.0
    gx, gy, gz = xs.mean(), ys.mean(), zs.mean()
    g = math.sqrt(gx * gx + gy * gy + gz * gz) + _EPS
    return {
        "mag_mean": float(mag.mean()), "mag_std": float(mag.std()),
        "energy": float((dyn ** 2).mean()), "band_ratio": band / total,
        "dom_freq": dom, "incl_deg": math.degrees(math.acos(max(-1.0, min(1.0, gz / g)))),
        "gx": float(gx / g), "gy": float(gy / g), "gz": float(gz / g),
    }


_IMU_KEYS = ("mag_mean", "mag_std", "energy", "band_ratio", "dom_freq",
             "incl_deg", "gx", "gy", "gz")


# ---------------------------------------------------------------------------
# Radar sector SNR (presence light model)
# ---------------------------------------------------------------------------

class SectorSNR:
    """Per-direction SNR of the RA view against an empty-room background."""

    def __init__(self, cfg: Dict[str, Any]):
        self.theta_s = float(cfg.get("theta_s_deg", 15.0))
        self.max_range_m = float(cfg.get("max_range_m", 3.0))
        self.near_rows = int(cfg.get("near_rows", 2))
        self.swap_lr = bool(cfg.get("swap_lr", False))
        self.background: Optional[np.ndarray] = None       # calibrated (24,24) log1p
        self._ring: Deque[np.ndarray] = deque(maxlen=int(cfg.get("bg_frames", 150)))
        self._bg_cache: Optional[np.ndarray] = None
        self._bg_age = 0

    def sector_masks(self) -> Dict[str, np.ndarray]:
        left = _ANGLES < -self.theta_s
        right = _ANGLES > self.theta_s
        if self.swap_lr:
            left, right = right, left
        return {"L": left, "M": ~(left | right), "R": right}

    def _bg(self) -> Optional[np.ndarray]:
        if self.background is not None:
            return self.background
        if len(self._ring) < 20:
            return None
        self._bg_age += 1
        if self._bg_cache is None or self._bg_age >= 10:
            self._bg_cache = np.median(np.stack(self._ring), axis=0)
            self._bg_age = 0
        return self._bg_cache

    def scores(self, ra: np.ndarray, learn: bool = True) -> Optional[Dict[str, Tuple[float, int]]]:
        """RA view → {sector: (snr_db, range_row)}; None while warming up."""
        if learn and self.background is None:
            self._ring.append(ra)
        bg = self._bg()
        if bg is None:
            return None
        snr = 10.0 * np.log10((np.expm1(ra) + 1e-3) / (np.expm1(bg) + 1e-3))
        snr[: self.near_rows, :] = -np.inf
        out = {}
        for name, cols in self.sector_masks().items():
            sub = snr[:, cols]
            idx = int(np.argmax(sub))
            row, _ = divmod(idx, sub.shape[1])
            out[name] = (float(sub.flat[idx]), row)
        return out

    def row_to_m(self, row: float) -> float:
        return float(row) / 23.0 * self.max_range_m


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------

class TaskModelHub:
    """Runs the active model per task and owns calibration sessions."""

    def __init__(self, state_dir: Path, config: Optional[Callable[[], Dict[str, Any]]] = None,
                 device_id: str = ""):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.device_id = device_id
        self._config = config or (lambda: {})
        self._lock = threading.RLock()
        self._active_path = self.state_dir / "models_active.json"
        self._cal_path = self.state_dir / "calibration.json"
        self.active: Dict[str, str] = {t: m[0] for t, m in TASKS.items()}
        self.active.update(self._read_json(self._active_path, {}))
        self.cal: Dict[str, Any] = self._read_json(self._cal_path, {})
        self.snr = SectorSNR(self._sensing_cfg())
        if self.cal.get("background"):
            self.snr.background = np.asarray(self.cal["background"], dtype=np.float32)
        self._frames: Deque[Tuple[float, Dict[str, Tuple[float, int]]]] = deque(maxlen=50)
        self._imu: Dict[str, Deque[Tuple[float, Dict[str, float]]]] = defaultdict(lambda: deque(maxlen=16))
        self._zone_clf = self._load_model("zone_clf")
        self._har_gbm = self._load_model("har_gbm")
        # Frozen radar-repo RF (E2 frozen selection, radar view): bundle
        # {model, scaler, fill, columns, threshold}. Seeded from
        # models/radar/presence_rf_radar.joblib on first start.
        self._rf_radar = self._load_model("presence_rf_radar") or self._seed_rf_radar()
        self._rf_frames: Deque[Tuple[float, np.ndarray, float]] = deque(maxlen=400)
        self._occ = None
        self._session: Optional[Dict[str, Any]] = None
        self.latest: Dict[str, Dict[str, Any]] = {}

    # -- persistence -----------------------------------------------------------
    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text())
        except Exception:
            return default

    def _write_json(self, path: Path, data: Any) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)

    def _load_model(self, name: str):
        path = self.state_dir / f"{name}.joblib"
        if not path.exists():
            return None
        try:
            import joblib
            return joblib.load(path)
        except Exception as exc:
            logger.warning("task model %s unreadable: %s", name, exc)
            return None

    def _seed_rf_radar(self):
        src = Path(__file__).resolve().parent.parent / "models" / "radar" / "presence_rf_radar.joblib"
        if not src.exists():
            return None
        try:
            import joblib
            obj = joblib.load(src)
            self._save_model("presence_rf_radar", obj)
            return obj
        except Exception as exc:
            logger.warning("presence-rf-radar seed unreadable: %s", exc)
            return None

    def _save_model(self, name: str, obj: Any) -> None:
        import joblib
        joblib.dump(obj, self.state_dir / f"{name}.joblib")

    def _sensing_cfg(self) -> Dict[str, Any]:
        cfg = (self._config() or {}).get("sensing") or {}
        return cfg if isinstance(cfg, dict) else {}

    # -- selection ---------------------------------------------------------------
    def selection(self) -> Dict[str, Any]:
        return {"active": dict(self.active),
                "options": {t: list(m) for t, m in TASKS.items()},
                "trained": {"zone-clf@1": self._zone_clf is not None,
                            "har-gbm@1": self._har_gbm is not None},
                "calibrated": {"presence-snr@1": bool(self.cal.get("thresholds")),
                               "zone-snr@1": "r_th_m" in self.cal}}

    def set_active(self, task: str, model: str) -> Dict[str, Any]:
        if task not in TASKS or model not in TASKS[task]:
            raise ValueError(f"unknown model {model!r} for task {task!r}")
        with self._lock:
            self.active[task] = model
            self._write_json(self._active_path, self.active)
        return self.selection()

    # -- window helpers ------------------------------------------------------------
    @staticmethod
    def _samples_by_type(window: Any) -> Tuple[List[Any], Dict[str, List[Any]]]:
        radar, csi = [], {}
        for sid, chunk in (getattr(window, "samples", None) or {}).items():
            for s in chunk or []:
                payload = getattr(s, "payload", None)
                enc = payload.get("encoding") if isinstance(payload, dict) else None
                stype = str(getattr(s, "sensor_type", "") or "")
                if enc == "radar_frame" or stype == "radar":
                    radar.append(s)
                elif (enc == "csi_raw" or stype == "csi"
                      or stype.startswith("wifi_csi")
                      or stype.startswith("radio.wifi_csi")):
                    csi.setdefault(sid, []).append(s)
        radar.sort(key=lambda s: s.timestamp)
        return radar, csi

    def _csi_sides(self, csi: Dict[str, List[Any]]) -> Dict[str, str]:
        configured = self._sensing_cfg().get("csi_sides") or {}
        sides: Dict[str, str] = {}
        for sid, chunk in csi.items():
            side = configured.get(sid)
            if not side and chunk and isinstance(chunk[-1].payload, dict):
                side = chunk[-1].payload.get("side")
            if side in ("L", "R"):
                sides[sid] = side
        rest = sorted(s for s in csi if s not in sides)
        for sid in rest:
            sides[sid] = "L" if "L" not in sides.values() else "R"
        return sides

    def _csi_energy(self, csi: Dict[str, List[Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"L": None, "R": None}
        for sid, side in self._csi_sides(csi).items():
            amps = [a for a in (csi_amplitude(s.payload) for s in csi[sid]) if a is not None]
            e = link_energy(amps)
            if not math.isnan(e):
                out[side] = e
        return out

    # -- per-tick inference ----------------------------------------------------------
    def step(self, window: Any, long_window: Optional[Callable[[], Any]] = None,
             now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Consume a synchronized window; return prediction dicts
        (``task``, ``label``, ``confidence``, ``model``, ``scores``, ``meta``)."""
        now = now or time.time()
        radar, csi = self._samples_by_type(window)
        for s in radar:
            views = decode_views(s.payload)
            if views is None:
                continue
            snr_f = s.payload.get("snr_f") if isinstance(s.payload, dict) else None
            self._rf_frames.append((float(s.timestamp), views,
                                    float(snr_f) if isinstance(snr_f, (int, float)) else float("nan")))
            sess = self._session
            if sess and sess["kind"] == "room" and sess.get("current") == "EMPTY":
                if len(sess["empty_views"]) < 1200:
                    sess["empty_views"].append(views[1])
            sc = self.snr.scores(views[1], learn=self._presence_label_hint() != "occupied")
            if sc is not None:
                self._frames.append((float(s.timestamp), sc))
        energy = self._csi_energy(csi)
        feats = self._radar_summary(now)
        self._record_room_sample(feats, energy, now)
        out: List[Dict[str, Any]] = []
        presence = self._run_presence(feats, energy, long_window, now)
        if presence:
            out.append(presence)
        zone = self._run_zone(feats, energy, presence, now)
        if zone:
            out.append(zone)
        for pred in out:
            self.latest[pred["task"]] = pred
        return out

    def _presence_label_hint(self) -> Optional[str]:
        p = self.latest.get("presence")
        return p.get("label") if p else None

    def _thresholds(self) -> Dict[str, float]:
        th = self.cal.get("thresholds") or {}
        default = float(self._sensing_cfg().get("default_snr_db", 6.0))
        return {d: float(th.get(d, default)) for d in SECTORS}

    # -- live sensing snapshot (room1 device screen) ------------------------------
    def zone_threshold(self, zone: str) -> float:
        """Occupied SNR threshold for a 6-section zone. Per-zone overrides
        (``cal["zone_thresholds"]``) fall back to the sector threshold."""
        if zone not in ZONES:
            raise ValueError(f"unknown zone {zone!r}")
        zt = self.cal.get("zone_thresholds") or {}
        if zone in zt:
            return float(zt[zone])
        return self._thresholds().get(zone[0], 6.0)

    def set_thresholds(self, sectors: Optional[Dict[str, float]] = None,
                       zones: Optional[Dict[str, float]] = None,
                       r_th_m: Optional[float] = None) -> Dict[str, Any]:
        """Easy threshold setting from the room1 screen — per-sector and/or
        per-zone occupied thresholds plus the near/far range boundary."""
        with self._lock:
            if sectors:
                th = self.cal.setdefault("thresholds", {})
                for d, v in sectors.items():
                    if d not in SECTORS:
                        raise ValueError(f"unknown sector {d!r}")
                    th[d] = float(v)
            if zones:
                zt = self.cal.setdefault("zone_thresholds", {})
                for z, v in zones.items():
                    if z not in ZONES:
                        raise ValueError(f"unknown zone {z!r}")
                    if v is None:
                        zt.pop(z, None)
                    else:
                        zt[z] = float(v)
            if r_th_m is not None:
                r = float(r_th_m)
                if not (0.05 <= r <= self.snr.max_range_m):
                    raise ValueError(f"r_th_m {r} out of range")
                self.cal["r_th_m"] = r
            self._write_json(self._cal_path, self.cal)
        return self.sensing_snapshot()

    def _zone_band_row(self, band: str) -> float:
        r_th = float(self.cal.get("r_th_m", self.snr.max_range_m / 2.0))
        return r_th / 2.0 if band == "NEAR" else (r_th + self.snr.max_range_m) / 2.0

    def sensing_snapshot(self, window_s: float = 2.5) -> Dict[str, Any]:
        """Live per-direction signal + per-zone occupancy for the room1
        device screen: sector SNR vs threshold, range band → 6 zones."""
        now = time.time()
        frames = [f for t, f in self._frames if now - t <= window_s][-10:]
        th = self._thresholds()
        sector_rows = {d: [f[d][1] for f in frames] for d in SECTORS}
        sectors: Dict[str, Any] = {}
        for d in SECTORS:
            snr = float(np.mean([f[d][0] for f in frames])) if frames else 0.0
            row = float(np.median(sector_rows[d])) if sector_rows[d] else 0.0
            sectors[d] = {"snr_db": round(snr, 2), "threshold_db": th[d],
                          "margin_db": round(snr - th[d], 2),
                          "range_m": round(self.snr.row_to_m(row), 2),
                          "occupied": bool(frames) and snr > th[d]}
        r_th = float(self.cal.get("r_th_m", self.snr.max_range_m / 2.0))
        zones: Dict[str, Any] = {}
        for z in ZONES:
            sector, band = z.split("_", 1)
            sec = sectors[sector]
            in_band = (sec["range_m"] <= r_th) == (band == "NEAR")
            zones[z] = {"occupied": bool(sec["occupied"] and in_band),
                        "snr_db": sec["snr_db"],
                        "threshold_db": self.zone_threshold(z),
                        "sector": sector, "band": band}
            # Per-zone override applies to the sector's margin check too.
            if self.cal.get("zone_thresholds") and z in self.cal["zone_thresholds"]:
                zones[z]["occupied"] = bool(in_band and sec["snr_db"] > zones[z]["threshold_db"])
        presence = self.latest.get("presence") or {}
        zone_pred = self.latest.get("zone") or {}
        return {"ts": now, "frames": len(frames),
                "sectors": sectors, "zones": zones, "r_th_m": r_th,
                "presence": presence.get("label"), "zone": zone_pred.get("label"),
                "active": dict(self.active)}

    def _radar_summary(self, now: float) -> Optional[Dict[str, Any]]:
        frames = [f for t, f in self._frames if now - t <= 2.5][-10:]
        if not frames:
            return None
        th = self._thresholds()
        margins = {d: float(np.mean([f[d][0] for f in frames])) - th[d] for d in SECTORS}
        exceed = [any(f[d][0] > th[d] for d in SECTORS) for f in frames]
        best = max(SECTORS, key=lambda d: margins[d])
        rows = [f[best][1] for f in frames]
        return {"frames": len(frames), "exceed": sum(exceed), "margins": margins,
                "snr": {d: float(np.mean([f[d][0] for f in frames])) for d in SECTORS},
                "best": best, "range_m": self.snr.row_to_m(float(np.median(rows)))}

    def _run_presence(self, feats, energy, long_window, now) -> Optional[Dict[str, Any]]:
        model = self.active["presence"]
        if model == "presence-occ-hier@2":
            return self._presence_heavy(long_window, now)
        if model == "presence-rf-radar@3":
            return self._presence_rf_radar(now)
        if feats is None:
            return {"task": "presence", "model": model, "label": "unknown", "confidence": 0.0,
                    "scores": {}, "meta": {"reason": "no radar frames"}, "ts": now}
        need = max(1, int(round(0.6 * feats["frames"])))
        occupied = feats["exceed"] >= need
        top = max(feats["margins"].values())
        conf = 1.0 / (1.0 + math.exp(-top / 3.0))
        calibrated = bool(self.cal.get("thresholds"))
        if not calibrated:
            conf = min(conf, 0.6) if occupied else min(1 - conf, 0.6)
        else:
            conf = conf if occupied else 1.0 - conf
        return {"task": "presence", "model": model,
                "label": "occupied" if occupied else "empty",
                "confidence": round(float(conf), 3),
                "scores": {d: round(m, 2) for d, m in feats["margins"].items()},
                "meta": {"direction": feats["best"] if occupied else None,
                         "snr_db": {d: round(v, 2) for d, v in feats["snr"].items()},
                         "thresholds_db": self._thresholds(), "calibrated": calibrated,
                         "frames": feats["frames"]},
                "ts": now}

    def _presence_rf_radar(self, now: float) -> Dict[str, Any]:
        model = "presence-rf-radar@3"
        base = {"task": "presence", "model": model, "ts": now}
        bundle = self._rf_radar
        if not bundle:
            return {**base, "label": "unknown", "confidence": 0.0, "scores": {},
                    "meta": {"reason": "model bundle missing"}}
        frames = [(v, snr) for t, v, snr in self._rf_frames if now - t <= 5.0]
        feats = radar_window_features(frames)
        if feats is None:
            return {**base, "label": "unknown", "confidence": 0.0, "scores": {},
                    "meta": {"reason": "need ≥5 radar frames in 5 s", "frames": len(frames)}}
        try:
            cols = list(bundle["columns"])
            fill = np.asarray(bundle["fill"], dtype=np.float64)
            x = np.asarray([feats.get(c, float("nan")) for c in cols], dtype=np.float64)
            bad = ~np.isfinite(x)
            x[bad] = fill[bad]
            p = float(bundle["model"].predict_proba(bundle["scaler"].transform(x[None, :]))[0, 1])
        except Exception as exc:
            logger.warning("presence-rf-radar failed: %s", exc)
            return {**base, "label": "unknown", "confidence": 0.0, "scores": {},
                    "meta": {"reason": f"model error: {type(exc).__name__}"}}
        thr = float(bundle.get("threshold", 0.6045833333333333))
        label = "occupied" if p >= thr else "empty"
        return {**base, "label": label,
                "confidence": round(p if label == "occupied" else 1 - p, 3),
                "scores": {"occupied": round(p, 3), "empty": round(1 - p, 3)},
                "meta": {"threshold": thr, "frames": len(frames)}}

    def _presence_heavy(self, long_window, now) -> Dict[str, Any]:
        model = "presence-occ-hier@2"
        base = {"task": "presence", "model": model, "ts": now}
        if long_window is None:
            return {**base, "label": "unknown", "confidence": 0.0, "scores": {},
                    "meta": {"reason": "no 15 s window available"}}
        try:
            if self._occ is None:
                from whispy_model_occ.hier import OccHierModel
                self._occ = OccHierModel({})
            window = long_window()
            radar, csi = self._samples_by_type(window)
            if not radar or not csi:
                return {**base, "label": "unknown", "confidence": 0.0, "scores": {},
                        "meta": {"reason": "needs radar + CSI", "radar": len(radar),
                                 "csi_links": len(csi)}}
            from whispy.models.runner import BINDINGS_KEY
            probs, details = [], []
            radar_sid = radar[0].sensor_id
            for sid in sorted(csi):
                window.preprocessing[BINDINGS_KEY] = {"csi": sid, "radar": radar_sid}
                p, detail = self._occ.proba(window)
                if p is not None:
                    probs.append(float(p))
                    details.append({"csi": sid, "p": round(float(p), 3)})
            if not probs:
                return {**base, "label": "unknown", "confidence": 0.0, "scores": {},
                        "meta": {"reason": "insufficient seconds in window"}}
            p = float(np.mean(probs))
            thr = float(self._occ.threshold if self._occ.threshold is not None else 0.62)
            label = "occupied" if p >= thr else "empty"
            return {**base, "label": label,
                    "confidence": round(p if label == "occupied" else 1 - p, 3),
                    "scores": {"occupied": round(p, 3), "empty": round(1 - p, 3)},
                    "meta": {"links": details, "threshold": thr}}
        except Exception as exc:
            logger.warning("presence heavy model failed: %s", exc)
            return {**base, "label": "unknown", "confidence": 0.0, "scores": {},
                    "meta": {"reason": f"model error: {type(exc).__name__}"}}

    def _band(self, range_m: float) -> str:
        r_th = float(self.cal.get("r_th_m", self.snr.max_range_m / 2.0))
        hyst = float(self._sensing_cfg().get("band_hysteresis_m", 0.25))
        prev = (self.latest.get("zone") or {}).get("label", "")
        if prev.endswith("_NEAR") and range_m < r_th + hyst:
            return "NEAR"
        if prev.endswith("_FAR") and range_m > r_th - hyst:
            return "FAR"
        return "NEAR" if range_m <= r_th else "FAR"

    def _zone_vector(self, feats, energy) -> Optional[List[float]]:
        if feats is None:
            return None
        el = energy.get("L") or 0.0
        er = energy.get("R") or 0.0
        return [feats["snr"]["L"], feats["snr"]["M"], feats["snr"]["R"], feats["range_m"],
                el, er, el / (el + er + _EPS)]

    def _run_zone(self, feats, energy, presence, now) -> Optional[Dict[str, Any]]:
        model = self.active["zone"]
        base = {"task": "zone", "model": model, "ts": now}
        if presence is None or presence["label"] == "unknown" or feats is None:
            return {**base, "label": "UNKNOWN", "confidence": 0.0, "scores": {}, "meta": {}}
        if presence["label"] == "empty":
            return {**base, "label": "EMPTY", "confidence": presence["confidence"],
                    "scores": {}, "meta": {}}
        if model == "zone-clf@1" and self._zone_clf is not None:
            vec = np.asarray([self._zone_vector(feats, energy)], dtype=np.float64)
            proba = self._zone_clf.predict_proba(vec)[0]
            classes = list(self._zone_clf.classes_)
            i = int(np.argmax(proba))
            label = classes[i] if proba[i] >= 0.4 else "UNKNOWN"
            return {**base, "label": label, "confidence": round(float(proba[i]), 3),
                    "scores": {c: round(float(p), 3) for c, p in zip(classes, proba)},
                    "meta": {"range_m": round(feats["range_m"], 2)}}
        direction = presence["meta"].get("direction") or feats["best"]
        band = self._band(feats["range_m"])
        meta = {"range_m": round(feats["range_m"], 2), "csi_energy": energy,
                "r_th_m": float(self.cal.get("r_th_m", self.snr.max_range_m / 2.0))}
        if model == "zone-clf@1":
            meta["fallback"] = "zone-clf@1 untrained — using zone-snr@1"
        return {**base, "label": f"{direction}_{band}",
                "confidence": round(float(presence["confidence"]), 3),
                "scores": feats["margins"], "meta": meta}

    # -- IMU / activity ---------------------------------------------------------------
    def on_observation(self, obs: Dict[str, Any], now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if (obs or {}).get("schema") != "imu.window.v1":
            return None
        f = imu_features(obs.get("value") or {})
        if f is None:
            return None
        now = now or float(obs.get("timestamp") or time.time())
        subject = str(obs.get("subject") or obs.get("source_id") or "unknown")
        self._imu[subject].append((now, f))
        self._record_activity_sample(subject, f, now)
        pred = self._run_activity(subject, f, now)
        self.latest[f"activity:{subject}"] = pred
        fused = self._fused_activity(now)
        if fused is not None:
            self.latest["activity:fused"] = fused
            return fused          # fused prediction is the better signal
        return pred

    def _fused_activity(self, now: float, window_s: float = 2.0) -> Optional[Dict[str, Any]]:
        """Watch + necklace IMU fusion: when ≥2 subjects report fresh
        windows, combine their features (mean) into one activity
        prediction — more robust than either device alone."""
        subjects = [s for s, dq in self._imu.items()
                    if dq and now - dq[-1][0] <= window_s]
        if len(subjects) < 2:
            return None
        feats = []
        for s in subjects:
            # Use the freshest feature row per subject.
            feats.append(self._imu[s][-1][1])
        f = {k: float(np.mean([x[k] for x in feats])) for k in _IMU_KEYS}
        pred = self._run_activity("fused", f, now)
        pred["meta"]["sources"] = sorted(subjects)
        pred["meta"]["fusion"] = "mean"
        # Confidence gets a small boost — two sensors agreeing.
        pred["confidence"] = round(min(1.0, pred.get("confidence", 0.0) + 0.1), 3)
        return pred

    def _rules(self) -> Dict[str, float]:
        r = {"still_std": 0.05, "walk_std_max": 0.6, "walk_band": 0.4,
             "run_std": 0.6, "run_freq": 2.2, "lying_incl": 60.0}
        r.update(self.cal.get("har_rules") or {})
        return r

    def _run_activity(self, subject: str, f: Dict[str, float], now: float) -> Dict[str, Any]:
        model = self.active["activity"]
        base = {"task": "activity", "model": model, "subject": subject, "ts": now}
        if model == "har-gbm@1" and self._har_gbm is not None:
            vec = np.asarray([[f[k] for k in _IMU_KEYS]], dtype=np.float64)
            proba = self._har_gbm["activity"].predict_proba(vec)[0]
            classes = list(self._har_gbm["activity"].classes_)
            i = int(np.argmax(proba))
            posture = None
            if self._har_gbm.get("posture") is not None:
                posture = str(self._har_gbm["posture"].predict(vec)[0])
            return {**base, "label": classes[i], "confidence": round(float(proba[i]), 3),
                    "scores": {c: round(float(p), 3) for c, p in zip(classes, proba)},
                    "meta": {"posture": posture}}
        r = self._rules()
        std, band, freq = f["mag_std"], f["band_ratio"], f["dom_freq"]
        if std < r["still_std"]:
            label, conf = "still", 0.8
        elif std > r["run_std"] and freq > r["run_freq"]:
            label, conf = "running", 0.7
        elif band > r["walk_band"] and std <= r["walk_std_max"]:
            label, conf = "walking", 0.7
        else:
            label, conf = "other", 0.4
        posture = "unknown"
        upright = self.cal.get("upright", {}).get(subject)
        incl = f["incl_deg"]
        if upright is not None:
            g = np.array([f["gx"], f["gy"], f["gz"]])
            u = np.asarray(upright, dtype=np.float64)
            incl = math.degrees(math.acos(float(np.clip(g @ u / (np.linalg.norm(u) + _EPS), -1, 1))))
        if label == "still" and incl > r["lying_incl"]:
            posture = "lying"
        elif label in ("walking", "running"):
            posture = "standing"
        meta = {"posture": posture, "features": {k: round(v, 4) for k, v in f.items()}}
        if model == "har-gbm@1":
            meta["fallback"] = "har-gbm@1 untrained — using har-rules@1"
        return {**base, "label": label, "confidence": conf, "scores": {}, "meta": meta}

    # -- calibration sessions ------------------------------------------------------------
    def calibration_status(self) -> Dict[str, Any]:
        s = self._session
        if not s:
            return {"active": False, "calibration": {k: v for k, v in self.cal.items() if k != "background"}}
        return {"active": True, "kind": s["kind"], "id": s["id"], "steps": s["steps"],
                "current": s.get("current"), "counts": {k: len(v) for k, v in s["data"].items()}}

    def start_calibration(self, kind: str, steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if kind not in ("room", "activity"):
            raise ValueError("kind must be 'room' or 'activity'")
        catalog = ROOM_STEPS if kind == "room" else ACTIVITY_STEPS
        if steps and all(isinstance(s, str) for s in steps):
            # Subset by label (e.g. the watch's stand/sit/sleep session).
            by_label = {st["label"]: st for st in catalog}
            unknown = [s for s in steps if s not in by_label]
            if unknown:
                raise ValueError(f"unknown calibration steps: {unknown}")
            steps = [dict(by_label[s], optional=False) for s in steps]
        with self._lock:
            self._session = {"kind": kind, "id": f"cal-{int(time.time())}",
                             "steps": steps or catalog,
                             "current": None, "data": defaultdict(list), "empty_views": [],
                             "started_at": time.time()}
        return self.calibration_status()

    def step_calibration(self, label: str, action: str) -> Dict[str, Any]:
        s = self._session
        if not s:
            raise ValueError("no calibration session")
        if label not in {st["label"] for st in s["steps"]}:
            raise ValueError(f"unknown step {label!r}")
        with self._lock:
            if action == "start":
                s["current"] = label
            elif action == "stop":
                s["current"] = None
                if s["kind"] == "room" and label == "EMPTY":
                    s["empty_report"] = self._apply_empty(s)
                    self._write_json(self._cal_path, self.cal)
            elif action == "retake":
                s["data"][label] = []
                s["current"] = label
            else:
                raise ValueError("action must be start|stop|retake")
        return self.calibration_status()

    def _record_room_sample(self, feats, energy, now) -> None:
        s = self._session
        if not s or s["kind"] != "room" or not s.get("current") or feats is None:
            return
        frames = [f for t, f in self._frames if now - t <= 2.5][-10:]
        s["data"][s["current"]].append({"vec": self._zone_vector(feats, energy),
                                        "frames": frames})

    def _record_activity_sample(self, subject: str, f: Dict[str, float], now: float) -> None:
        s = self._session
        if not s or s["kind"] != "activity" or not s.get("current"):
            return
        s["data"][s["current"]].append({"subject": subject, "f": f, "ts": now})

    def finish_calibration(self) -> Dict[str, Any]:
        s = self._session
        if not s:
            raise ValueError("no calibration session")
        with self._lock:
            report = (self._finish_room(s) if s["kind"] == "room" else self._finish_activity(s))
            self._session = None
            self._write_json(self._cal_path, self.cal)
        return report

    def _apply_empty(self, s: Dict[str, Any]) -> Dict[str, Any]:
        """EMPTY step → calibrated background + per-sector SNR thresholds.

        Applied as soon as the EMPTY step stops so the zone steps that
        follow are scored against the same background used at inference."""
        views = s.get("empty_views") or []
        if len(views) < 20:
            return {"error": f"EMPTY step captured {len(views)} radar frames (need ≥ 20)"}
        self.snr.background = np.median(np.stack(views), axis=0).astype(np.float32)
        self.cal["background"] = self.snr.background.round(4).tolist()
        scored = [self.snr.scores(v, learn=False) for v in views]
        k = float(self._sensing_cfg().get("threshold_k", 3.0))
        floor = float(self._sensing_cfg().get("min_snr_db", 3.0))
        self.cal["thresholds"] = {
            d: round(max(floor, float(np.mean([x[d][0] for x in scored])
                                      + k * np.std([x[d][0] for x in scored]))), 3)
            for d in SECTORS}
        self._frames.clear()
        return {"background_frames": len(views), "thresholds_db": self.cal["thresholds"]}

    def _finish_room(self, s: Dict[str, Any]) -> Dict[str, Any]:
        report: Dict[str, Any] = {"kind": "room", "id": s["id"]}
        if "empty_report" not in s:
            s["empty_report"] = self._apply_empty(s)
        report.update(s["empty_report"])
        near = [r["vec"][3] for z in ZONES if z.endswith("NEAR") for r in s["data"].get(z, [])]
        far = [r["vec"][3] for z in ZONES if z.endswith("FAR") for r in s["data"].get(z, [])]
        if near and far:
            cands = np.linspace(min(near + far), max(near + far), 50)
            best = max(cands, key=lambda t: 0.5 * (np.mean(np.array(near) <= t)
                                                    + np.mean(np.array(far) > t)))
            self.cal["r_th_m"] = round(float(best), 3)
            report["r_th_m"] = self.cal["r_th_m"]
        X, y = [], []
        for z in ZONES:
            for r in s["data"].get(z, []):
                if r["vec"] is not None:
                    X.append(r["vec"])
                    y.append(z)
        report["samples"] = {z: len(s["data"].get(z, [])) for z in ("EMPTY",) + ZONES}
        if len(set(y)) >= 2 and len(X) >= 12:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))
            clf.fit(np.asarray(X), np.asarray(y))
            self._zone_clf = clf
            self._save_model("zone_clf", clf)
            report["zone_clf_train_accuracy"] = round(float(clf.score(np.asarray(X), np.asarray(y))), 3)
        self.cal["room_calibrated_at"] = time.time()
        return report

    def _finish_activity(self, s: Dict[str, Any]) -> Dict[str, Any]:
        steps = {st["label"]: st for st in s["steps"]}
        X, ya, yp = [], [], []
        upright: Dict[str, List[List[float]]] = defaultdict(list)
        for label, rows in s["data"].items():
            st = steps.get(label, {})
            for r in rows:
                X.append([r["f"][k] for k in _IMU_KEYS])
                ya.append(st.get("activity", label))
                yp.append(st.get("posture", "unknown"))
                if st.get("posture") == "standing" and st.get("activity") == "still":
                    upright[r["subject"]].append([r["f"]["gx"], r["f"]["gy"], r["f"]["gz"]])
        report: Dict[str, Any] = {"kind": "activity", "id": s["id"],
                                  "samples": {k: len(v) for k, v in s["data"].items()}}
        if upright:
            self.cal.setdefault("upright", {}).update(
                {subj: np.mean(v, axis=0).round(4).tolist() for subj, v in upright.items()})
        still = [x for x, a in zip(X, ya) if a == "still"]
        walk = [x for x, a in zip(X, ya) if a == "walking"]
        if still and walk:
            std_i = _IMU_KEYS.index("mag_std")
            still_hi = float(np.percentile([x[std_i] for x in still], 95))
            walk_lo = float(np.percentile([x[std_i] for x in walk], 5))
            self.cal.setdefault("har_rules", {})["still_std"] = round((still_hi + walk_lo) / 2, 4)
            report["har_rules"] = self.cal["har_rules"]
        if len(set(ya)) >= 2 and len(X) >= 10:
            from sklearn.ensemble import HistGradientBoostingClassifier
            Xa = np.asarray(X, dtype=np.float64)
            act = HistGradientBoostingClassifier(max_iter=150).fit(Xa, np.asarray(ya))
            post = None
            if len(set(yp)) >= 2:
                post = HistGradientBoostingClassifier(max_iter=150).fit(Xa, np.asarray(yp))
            self._har_gbm = {"activity": act, "posture": post}
            self._save_model("har_gbm", self._har_gbm)
            report["har_gbm_train_accuracy"] = round(float(act.score(Xa, np.asarray(ya))), 3)
        self.cal["activity_calibrated_at"] = time.time()
        return report
