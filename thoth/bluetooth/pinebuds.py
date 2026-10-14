"""PineBuds Pro (thothBuds firmware) link — central-role status/gesture
subscription and voice-session feedback.

The buds expose the Thoth wearable service ``thoth.wear/v1``
(``a06302xx-01c4-4d1e-9b2f-0011778899aa``, see
``architecture/wearables-integration.md`` §3). Every characteristic is
auth-gated on the firmware side, so the node must be *bonded* with the
buds — ``central_connect`` goes through BlueZ, which pairs on the first
``insufficient authentication`` ATT error when the adapter is set
pairable (``bluetoothctl pairable on``; agent ``NoInputNoOutput``).

Emits:

* ``wearable.status.v1`` — battery / link flags / session state, on
  change (≤1 Hz by firmware contract).
* ``wearable.gesture.v1`` — one observation per gesture mapped to a
  Thoth action; also forwarded as a ``wearable_gesture`` node event so
  Brain automations can react without polling evidence.
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ..observations import Observation

logger = logging.getLogger(__name__)

_BASE = "-01c4-4d1e-9b2f-0011778899aa"
SERVICE = "a0630200" + _BASE
CHAR_INFO = "a0630201" + _BASE       # READ
CHAR_STATUS = "a0630202" + _BASE     # READ + NOTIFY
CHAR_EVENT = "a0630203" + _BASE      # NOTIFY
CHAR_CONTROL = "a0630204" + _BASE    # WRITE

PROTO = 1

# INFO.caps bits
CAP_GESTURE_EVENTS = 1 << 0
CAP_BATTERY = 1 << 1
CAP_SESSION_FEEDBACK = 1 << 2
CAP_ANC = 1 << 3
CAP_TWS_PEER_STATUS = 1 << 4
CAP_MIC_HFP = 1 << 5

# STATUS.flags bits
FLAG_TWS_LINKED = 1 << 0
FLAG_A2DP_STREAMING = 1 << 1
FLAG_SCO_CALL = 1 << 2
FLAG_ANC_ON = 1 << 3
FLAG_CHARGING = 1 << 4
FLAG_ROLE_MASTER = 1 << 5

SESSION_STATES = {0: "idle", 1: "listening", 2: "processing", 3: "speaking"}
SIDES = {0: None, 1: "left", 2: "right"}
GESTURES = {1: "single_tap", 2: "double_tap", 3: "triple_tap",
            4: "long_press"}

# opb_button_action_t — stock codes + Thoth additions (0x0010+).
ACTIONS = {
    0x0000: "none", 0x0001: "play_pause", 0x0002: "next_track",
    0x0003: "previous_track", 0x0004: "volume_up", 0x0005: "volume_down",
    0x0006: "toggle_anc", 0x0007: "voice_assistant", 0x0008: "answer_call",
    0x0009: "reject_call", 0x000A: "end_call", 0x000B: "mute_mic",
    0x000C: "transparency", 0x000D: "anc_off", 0x000E: "custom_1",
    0x000F: "custom_2",
    0x0010: "thoth_voice_toggle", 0x0011: "thoth_voice_end",
    0x0012: "thoth_context", 0x0013: "thoth_automation_1",
    0x0014: "thoth_automation_2", 0x0015: "thoth_automation_3",
}
THOTH_ACTION_MIN = 0x0010

# CONTROL opcodes
OP_SESSION_STATE = 0x01
OP_EVENT_MASK = 0x02
OP_EPOCH = 0x03


@dataclass
class BudsInfo:
    proto: int
    firmware: str
    side: Optional[str]
    caps: int

    def capabilities(self) -> Dict[str, bool]:
        return {
            "gesture_events": bool(self.caps & CAP_GESTURE_EVENTS),
            "battery": bool(self.caps & CAP_BATTERY),
            "session_feedback": bool(self.caps & CAP_SESSION_FEEDBACK),
            "anc": bool(self.caps & CAP_ANC),
            "tws_peer_status": bool(self.caps & CAP_TWS_PEER_STATUS),
            "mic_hfp": bool(self.caps & CAP_MIC_HFP),
        }


def parse_info(data: bytes) -> Optional[BudsInfo]:
    """INFO: ``<BBBBBHB`` proto, fw major/minor/patch, side, caps, rsv."""
    if len(data) < 8:
        return None
    proto, ma, mi, pa, side, caps, _rsv = struct.unpack_from("<BBBBBHB",
                                                             data, 0)
    return BudsInfo(proto=proto, firmware=f"{ma}.{mi}.{pa}",
                    side=SIDES.get(side), caps=caps)


def parse_status(data: bytes) -> Optional[Dict[str, Any]]:
    """STATUS: ``<BIBBBB`` seq, tick_ms, batt_local, batt_peer, flags,
    session → JSON-safe dict."""
    if len(data) < 9:
        return None
    seq, tick_ms, bl, bp, flags, session = struct.unpack_from("<BIBBBB",
                                                              data, 0)
    return {
        "seq": seq,
        "tick_ms": tick_ms,
        "battery_pct": bl if bl <= 100 else None,
        "peer_battery_pct": bp if bp <= 100 else None,
        "tws_linked": bool(flags & FLAG_TWS_LINKED),
        "a2dp_streaming": bool(flags & FLAG_A2DP_STREAMING),
        "sco_call": bool(flags & FLAG_SCO_CALL),
        "anc_on": bool(flags & FLAG_ANC_ON),
        "charging": bool(flags & FLAG_CHARGING),
        "role_master": bool(flags & FLAG_ROLE_MASTER),
        "session": SESSION_STATES.get(session, f"unknown_{session}"),
    }


def parse_event(data: bytes) -> Optional[Dict[str, Any]]:
    """EVENT: ``<BIBBH`` seq, tick_ms, side, gesture, action."""
    if len(data) < 9:
        return None
    seq, tick_ms, side, gesture, action = struct.unpack_from("<BIBBH",
                                                             data, 0)
    return {
        "seq": seq,
        "tick_ms": tick_ms,
        "side": SIDES.get(side),
        "gesture": GESTURES.get(gesture, f"gesture_{gesture}"),
        "action": ACTIONS.get(action, f"action_{action:#06x}"),
        "action_code": action,
        "thoth_action": action >= THOTH_ACTION_MIN,
    }


def encode_session_state(state: str) -> bytes:
    code = {v: k for k, v in SESSION_STATES.items()}.get(state)
    if code is None:
        raise ValueError(f"unknown session state {state!r}")
    return bytes([OP_SESSION_STATE, code])


def encode_event_mask(all_gestures: bool) -> bytes:
    return bytes([OP_EVENT_MASK, 1 if all_gestures else 0])


def encode_epoch(epoch_ms: Optional[int] = None) -> bytes:
    if epoch_ms is None:
        epoch_ms = int(time.time() * 1000)
    return struct.pack("<BQ", OP_EPOCH, int(epoch_ms))


class PinebudsLink:
    """One enrolled pair of buds: status/gesture subscribe + session
    feedback writes, driven by a managed central session."""

    def __init__(self, ble: Any, address: str, subject: str,
                 emit: Callable[[Observation], Any], source_id: str,
                 emit_event: Optional[Callable[[str, Dict[str, Any]],
                                               Any]] = None):
        self._ble = ble
        self.address = address
        self.subject = subject
        self._emit = emit
        self._emit_event = emit_event
        self.source_id = source_id
        self._attached = False
        self._up = False
        self._seq = 0
        self._gesture_seq = 0
        self._last_status: Optional[Dict[str, Any]] = None
        self._last_event_seq: Optional[int] = None
        self.info: Optional[BudsInfo] = None
        self.dropped_events = 0

    # -- lifecycle ------------------------------------------------------------------
    def attach(self) -> bool:
        if self._attached:
            return True
        ok = self._ble.central_connect(
            self.address,
            subscriptions={
                CHAR_STATUS: self._on_status,
                CHAR_EVENT: self._on_event,
            },
            on_state=self._on_state)
        self._attached = ok
        return ok

    def detach(self) -> None:
        self._attached = False
        self._ble.central_disconnect(self.address)

    @property
    def connected(self) -> bool:
        return self._up

    def _on_state(self, address: str, up: bool) -> None:
        self._up = up
        if up:
            logger.info("earbuds link up: %s", self.subject)
            # Gateway time reference — logs only on the buds, but it
            # lets a UART trace line up with Brain timestamps.
            self._ble.central_write(self.address, CHAR_CONTROL,
                                    encode_epoch())
        else:
            self._last_event_seq = None

    # -- status ------------------------------------------------------------------------
    def _on_status(self, data: bytes) -> None:
        st = parse_status(data)
        if st is None:
            return
        snapshot = {k: v for k, v in st.items()
                    if k not in ("seq", "tick_ms")}
        if snapshot == self._last_status:
            return
        self._last_status = snapshot
        self._seq += 1
        try:
            self._emit(Observation(
                schema="wearable.status.v1",
                source_id=self.source_id,
                subject=self.subject,
                value=dict(st, kind="earbuds", model="pinebuds-pro"),
                units={"battery_pct": "%", "peer_battery_pct": "%"},
                sequence=self._seq,
                timestamp=time.time()).validate())
        except Exception as exc:
            logger.debug("earbuds status emit failed: %s", exc)

    # -- gestures ---------------------------------------------------------------------------
    def _on_event(self, data: bytes) -> None:
        ev = parse_event(data)
        if ev is None:
            return
        if self._last_event_seq is not None:
            gap = (ev["seq"] - self._last_event_seq - 1) & 0xFF
            if 0 < gap < 128:
                self.dropped_events += gap
        self._last_event_seq = ev["seq"]
        now = time.time()
        self._gesture_seq += 1
        value = dict(ev, kind="earbuds", model="pinebuds-pro")
        try:
            self._emit(Observation(
                schema="wearable.gesture.v1",
                source_id=self.source_id,
                subject=self.subject,
                value=value,
                sequence=self._gesture_seq,
                timestamp=now).validate())
        except Exception as exc:
            logger.debug("earbuds gesture emit failed: %s", exc)
        if self._emit_event is not None and ev["thoth_action"]:
            try:
                self._emit_event("wearable_gesture",
                                 dict(value, subject=self.subject,
                                      timestamp=now))
            except Exception as exc:
                logger.debug("earbuds gesture event failed: %s", exc)

    # -- control ------------------------------------------------------------------------------
    def set_session_state(self, state: str) -> bool:
        """Voice-session feedback: idle/listening/processing/speaking."""
        return bool(self._ble.central_write(self.address, CHAR_CONTROL,
                                            encode_session_state(state)))

    def set_event_mask(self, all_gestures: bool) -> bool:
        return bool(self._ble.central_write(self.address, CHAR_CONTROL,
                                            encode_event_mask(all_gestures)))

    def status(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "connected": self._up,
            "info": None if self.info is None else {
                "proto": self.info.proto,
                "firmware": self.info.firmware,
                "side": self.info.side,
                "capabilities": self.info.capabilities(),
            },
            "last_status": self._last_status,
            "dropped_events": self.dropped_events,
        }


__all__ = [
    "ACTIONS", "CHAR_CONTROL", "CHAR_EVENT", "CHAR_INFO", "CHAR_STATUS",
    "GESTURES", "PROTO", "PinebudsLink", "SERVICE", "SESSION_STATES",
    "BudsInfo", "encode_epoch", "encode_event_mask",
    "encode_session_state", "parse_event", "parse_info", "parse_status",
]
