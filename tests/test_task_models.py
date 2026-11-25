"""TaskModelHub — two models per task, switching, calibration, inference."""
import base64
import time

import numpy as np
import pytest

from thoth.task_models import (TASKS, ZONES, SectorSNR, TaskModelHub,
                               csi_amplitude, imu_features, link_energy)


# -- fixtures -----------------------------------------------------------------

def _views(snr_db: float = 0.0, sector: str = "M", row: int = 10):
    """(4,24,24) float16 views with an RA hotspot in one sector."""
    v = np.zeros((4, 24, 24), dtype=np.float16)
    if snr_db:
        # col ranges for L/M/R over the -60..+60 angle grid
        cols = {"L": slice(0, 6), "M": slice(9, 15), "R": slice(18, 24)}[sector]
        ra = v[1].astype(np.float32)
        ra[row, cols] = np.log1p(np.exp(snr_db / 10 * np.log(10)) - 1 + 0.1)
        v[1] = ra.astype(np.float16)
    return v


def _radar_sample(views, ts=None, sid="radar-1"):
    from types import SimpleNamespace
    return SimpleNamespace(
        sensor_id=sid, sensor_type="radar",
        timestamp=ts or time.time(),
        payload={"encoding": "radar_frame",
                 "views_b64": base64.b64encode(views.tobytes()).decode(),
                 "snr_f": 0.0})


def _window(samples):
    from types import SimpleNamespace
    by_sid = {}
    for s in samples:
        by_sid.setdefault(s.sensor_id, []).append(s)
    return SimpleNamespace(samples=by_sid, preprocessing={})


def _csi_sample(n=200, seed=0, sid="esp-left", ts=None):
    rng = np.random.default_rng(seed)
    iq = (rng.normal(0, 30, 52) + 1j * rng.normal(0, 30, 52))
    iq_list = []
    for c in iq:
        iq_list += [int(c.real), int(c.imag)]
    from types import SimpleNamespace
    return SimpleNamespace(sensor_id=sid, sensor_type="wifi_csi",
                           timestamp=ts or time.time(),
                           payload={"encoding": "csi_raw", "iq": iq_list})


@pytest.fixture
def hub(tmp_path):
    return TaskModelHub(tmp_path / "tm", config=lambda: {"sensing": {}},
                        device_id="room1-test")


# -- feature extraction ---------------------------------------------------------

def test_decode_views_and_sector_snr(hub):
    """A hotspot in the R sector must outscore L and M."""
    bg = _views()
    for _ in range(25):
        hub.snr.scores(bg[1])                       # learn empty background
    sc = hub.snr.scores(_views(12.0, "R", row=15)[1])
    assert sc is not None
    assert sc["R"][0] > sc["L"][0] + 5
    assert sc["R"][0] > sc["M"][0] + 5
    assert sc["R"][1] == 15                          # peak range row kept


def test_csi_amplitude_and_link_energy():
    s = _csi_sample()
    amps = csi_amplitude(s.payload)
    assert amps is not None and len(amps) == 52
    # steady link → low temporal variation; bursty → high
    steady = link_energy([np.full(52, 100.0)] * 20)
    bursty = link_energy([np.full(52, v) for v in
                          np.linspace(50, 150, 20)])
    assert steady < 0.01 < bursty


def test_imu_features_units_and_inclination():
    flat = {"rate_hz": 50, "axes": {"x": [0.0] * 32, "y": [0.0] * 32,
                                    "z": [1.0] * 32}}
    f = imu_features(flat)
    assert f["incl_deg"] < 5.0                       # upright z-axis
    lying = {"rate_hz": 50, "axes": {"x": [1.0] * 32, "y": [0.0] * 32,
                                     "z": [0.0] * 32}}
    assert imu_features(lying)["incl_deg"] > 80.0
    ms2 = {"rate_hz": 50, "axes": {"x": [0.0] * 32, "y": [0.0] * 32,
                                   "z": [9.81] * 32}}  # m/s² auto-normalized
    assert abs(imu_features(ms2)["mag_mean"] - 1.0) < 0.05


# -- model selection ------------------------------------------------------------

def test_selection_defaults_and_switching(hub):
    sel = hub.selection()
    assert sel["active"] == {"presence": "presence-snr@1",
                             "zone": "zone-snr@1", "activity": "har-rules@1"}
    hub.set_active("zone", "zone-clf@1")
    assert hub.active["zone"] == "zone-clf@1"
    with pytest.raises(ValueError):
        hub.set_active("zone", "bogus@9")
    with pytest.raises(ValueError):
        hub.set_active("bogus", "presence-snr@1")
    # persisted across restarts
    hub2 = TaskModelHub(hub.state_dir, config=lambda: {"sensing": {}})
    assert hub2.active["zone"] == "zone-clf@1"


# -- presence / zone inference ------------------------------------------------------

def _feed(hub, views_list, n_ticks=1, ts=None):
    out = []
    ts = ts or time.time()
    for v in views_list:
        out = hub.step(_window([_radar_sample(v, ts)]), now=ts)
    return out


def test_presence_unknown_then_empty(hub):
    # fewer than 20 warmup frames → background not yet learned
    preds = _feed(hub, [_views()] * 5)
    pres = [p for p in preds if p["task"] == "presence"][0]
    assert pres["label"] in ("empty", "unknown")     # never crashes pre-warmup


def test_presence_and_zone_occupied(hub):
    _feed(hub, [_views()] * 30)                      # learn empty room
    preds = _feed(hub, [_views(14.0, "L", row=6)] * 10)
    pres = [p for p in preds if p["task"] == "presence"][0]
    assert pres["label"] == "occupied"
    assert pres["meta"]["direction"] == "L"
    zone = [p for p in preds if p["task"] == "zone"][0]
    assert zone["label"].startswith("L_")
    assert zone["label"].endswith(("NEAR", "FAR"))


def test_zone_empty_when_presence_empty(hub):
    _feed(hub, [_views()] * 30)
    preds = _feed(hub, [_views()] * 10)
    pres = [p for p in preds if p["task"] == "presence"][0]
    zone = [p for p in preds if p["task"] == "zone"][0]
    if pres["label"] == "empty":
        assert zone["label"] == "EMPTY"


def test_heavy_presence_graceful_without_deps(hub, monkeypatch):
    """occ-hier active but radar/CSI absent → unknown, not an exception."""
    hub.set_active("presence", "presence-occ-hier@2")
    preds = hub.step(_window([_radar_sample(_views())]),
                     long_window=lambda: _window([_radar_sample(_views())]))
    pres = [p for p in preds if p["task"] == "presence"][0]
    assert pres["label"] == "unknown"


# -- activity ------------------------------------------------------------------------

def _imu_obs(axes, subject="watch-1"):
    return {"schema": "imu.window.v1", "subject": subject,
            "source_id": subject, "timestamp": time.time(),
            "value": {"rate_hz": 50, "axes": axes}}


def test_activity_rules_still_and_walk(hub):
    still = hub.on_observation(_imu_obs(
        {"x": [0.0] * 64, "y": [0.0] * 64, "z": [1.0] * 64}))
    assert still["label"] == "still"
    assert still["meta"]["posture"] == "unknown"     # no upright calibration
    t = np.arange(64)
    walk = hub.on_observation(_imu_obs(
        {"x": 0.35 * np.sin(2 * np.pi * 2.0 * t / 50), "y": [0.0] * 64,
         "z": 1.0 + 0.15 * np.sin(2 * np.pi * 2.0 * t / 50)}))
    assert walk["label"] in ("walking", "other")


def test_activity_gbm_fallback_when_untrained(hub):
    hub.set_active("activity", "har-gbm@1")
    pred = hub.on_observation(_imu_obs(
        {"x": [0.0] * 64, "y": [0.0] * 64, "z": [1.0] * 64}))
    assert "fallback" in pred["meta"]                # honest about rules use
    assert pred["label"] == "still"


# -- calibration ------------------------------------------------------------------

def test_room_calibration_flow(hub):
    hub.start_calibration("room")
    st = hub.calibration_status()
    assert st["active"] and st["kind"] == "room"
    labels = [s["label"] for s in st["steps"]]
    assert labels == ["EMPTY"] + list(ZONES)

    # EMPTY: feed background frames while the step is current
    hub.step_calibration("EMPTY", "start")
    _feed(hub, [_views()] * 30)
    rep = hub.step_calibration("EMPTY", "stop")
    assert rep["current"] is None
    assert "L" in hub.cal["thresholds"]

    # a couple of labelled zone windows
    hub.step_calibration("L_NEAR", "start")
    _feed(hub, [_views(14.0, "L", row=5)] * 8)
    hub.step_calibration("L_NEAR", "stop")
    hub.step_calibration("R_FAR", "start")
    _feed(hub, [_views(14.0, "R", row=18)] * 8)
    hub.step_calibration("R_FAR", "stop")

    report = hub.finish_calibration()
    assert report["kind"] == "room"
    assert report["samples"]["EMPTY"] >= 1 or report.get("background_frames")
    assert "r_th_m" in hub.cal
    assert hub.calibration_status()["active"] is False
    # background persisted and reloaded
    hub2 = TaskModelHub(hub.state_dir, config=lambda: {"sensing": {}})
    assert hub2.snr.background is not None
    assert "thresholds" in hub2.cal


def test_room_calibration_rejects_short_empty(hub):
    hub.start_calibration("room")
    hub.step_calibration("EMPTY", "start")
    _feed(hub, [_views()] * 3)
    rep = hub.step_calibration("EMPTY", "stop")
    assert "error" in (hub._session.get("empty_report") or {})


def test_activity_calibration_flow(hub):
    hub.start_calibration("activity")
    hub.step_calibration("standing", "start")
    for _ in range(12):
        hub.on_observation(_imu_obs(
            {"x": [0.0] * 64, "y": [0.0] * 64, "z": [1.0] * 64}))
    hub.step_calibration("standing", "stop")
    hub.step_calibration("walking", "start")
    t = np.arange(64)
    for _ in range(12):
        hub.on_observation(_imu_obs(
            {"x": 0.4 * np.sin(2 * np.pi * 2.0 * t / 50), "y": [0.0] * 64,
             "z": 1.0 + 0.2 * np.sin(2 * np.pi * 2.0 * t / 50)}))
    hub.step_calibration("walking", "stop")
    report = hub.finish_calibration()
    assert report["samples"]["walking"] == 12
    assert "watch-1" in hub.cal.get("upright", {})   # posture reference


def test_calibration_guards(hub):
    with pytest.raises(ValueError):
        hub.finish_calibration()
    with pytest.raises(ValueError):
        hub.step_calibration("EMPTY", "start")
    with pytest.raises(ValueError):
        hub.start_calibration("bogus")
    hub.start_calibration("room")
    with pytest.raises(ValueError):
        hub.step_calibration("NOPE", "start")
    with pytest.raises(ValueError):
        hub.step_calibration("EMPTY", "bogus")


# -- presence-rf-radar@3 (frozen radar-repo RF) ---------------------------------

def test_radar_window_features_match_repo_aggregates():
    from thoth.task_models import radar_window_features, RADAR_VIEW_NAMES
    rng = np.random.default_rng(0)
    frames = [(rng.random((4, 24, 24)).astype(np.float32), float(i)) for i in range(10)]
    feats = radar_window_features(frames)
    v = np.stack([f[0] for f in frames]).astype(np.float64)
    # radar/E1: S/n, sqrt(Q/n - mean^2), D/(n-1)
    S, Q, n = v.sum(0), (v ** 2).sum(0), len(frames)
    D = np.abs(np.diff(v, axis=0)).sum(0)
    mean, std = S / n, np.sqrt(np.clip(Q / n - (S / n) ** 2, 0, None))
    for i, name in enumerate(RADAR_VIEW_NAMES):
        assert feats[f"{name}_mean"] == pytest.approx(mean[i].mean())
        assert feats[f"{name}_p90"] == pytest.approx(np.quantile(mean[i], 0.9))
        assert feats[f"{name}_std90"] == pytest.approx(np.quantile(std[i], 0.9))
        assert feats[f"{name}_delta"] == pytest.approx((D[i] / (n - 1)).mean())
        assert feats[f"{name}_peak"] == pytest.approx(mean[i].max())
    assert feats["snr_mean"] == pytest.approx(4.5) and feats["snr_max"] == 9.0
    assert radar_window_features(frames[:4]) is None


def test_presence_rf_radar_runs_frozen_bundle(tmp_path):
    pytest.importorskip("sklearn")
    hub = TaskModelHub(tmp_path)
    if hub._rf_radar is None:
        pytest.skip("frozen bundle not loadable here")
    assert "presence-rf-radar@3" in TASKS["presence"]
    hub.set_active("presence", "presence-rf-radar@3")
    now = time.time()
    samples = [_radar_sample(_views(12.0), ts=now - 4 + i * 0.5) for i in range(8)]
    out = hub.step(_window(samples), now=now)
    pres = next(p for p in out if p["task"] == "presence")
    assert pres["model"] == "presence-rf-radar@3"
    assert pres["label"] in ("occupied", "empty")
    assert 0.0 <= pres["scores"]["occupied"] <= 1.0
