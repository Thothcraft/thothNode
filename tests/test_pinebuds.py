"""PineBuds (thothBuds) link — wire codecs + observation/event emission."""
import struct

from thoth.bluetooth import pinebuds as pb


class _Ble:
    source_id = "ble:hci0"

    def __init__(self):
        self.subs = {}
        self.writes = []
        self.on_state = None

    def central_connect(self, address, subscriptions=None, on_state=None):
        self.subs = dict(subscriptions or {})
        self.on_state = on_state
        return True

    def central_disconnect(self, address):
        return True

    def central_write(self, address, char_uuid, data):
        self.writes.append((char_uuid, bytes(data)))
        return True


def _status(seq=1, tick=1000, bl=87, bp=85, flags=0, session=0):
    return struct.pack("<BIBBBB", seq, tick, bl, bp, flags, session)


def _event(seq=1, tick=2000, side=2, gesture=2, action=0x0010):
    return struct.pack("<BIBBH", seq, tick, side, gesture, action)


def test_parse_info_caps_and_firmware():
    data = struct.pack("<BBBBBHB", 1, 1, 4, 0, 1,
                       pb.CAP_GESTURE_EVENTS | pb.CAP_BATTERY, 0)
    info = pb.parse_info(data)
    assert info.proto == 1 and info.firmware == "1.4.0"
    assert info.side == "left"
    caps = info.capabilities()
    assert caps["gesture_events"] and caps["battery"] and not caps["anc"]
    assert pb.parse_info(b"\x01\x02") is None


def test_parse_status_flags_and_unknown_peer():
    st = pb.parse_status(_status(bp=0xFF,
                                 flags=pb.FLAG_TWS_LINKED | pb.FLAG_SCO_CALL,
                                 session=1))
    assert st["battery_pct"] == 87 and st["peer_battery_pct"] is None
    assert st["tws_linked"] and st["sco_call"] and not st["anc_on"]
    assert st["session"] == "listening"


def test_parse_event_maps_thoth_actions():
    ev = pb.parse_event(_event())
    assert ev["side"] == "right" and ev["gesture"] == "double_tap"
    assert ev["action"] == "thoth_voice_toggle" and ev["thoth_action"]
    stock = pb.parse_event(_event(action=0x0001))
    assert stock["action"] == "play_pause" and not stock["thoth_action"]


def test_control_encoders():
    assert pb.encode_session_state("listening") == b"\x01\x01"
    assert pb.encode_event_mask(True) == b"\x02\x01"
    raw = pb.encode_epoch(1700000000123)
    assert raw[0] == pb.OP_EPOCH and len(raw) == 9
    assert struct.unpack_from("<Q", raw, 1)[0] == 1700000000123
    try:
        pb.encode_session_state("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_link_emits_status_once_per_change_and_gesture_events():
    ble = _Ble()
    seen, events = [], []
    link = pb.PinebudsLink(ble, "AA:BB:CC:DD:EE:FF", "device:buds", seen.append,
                           ble.source_id,
                           emit_event=lambda k, d: events.append((k, d)))
    assert link.attach()
    assert set(ble.subs) == {pb.CHAR_STATUS, pb.CHAR_EVENT}

    ble.on_state("AA:BB:CC:DD:EE:FF", True)
    assert link.connected
    assert ble.writes and ble.writes[0][0] == pb.CHAR_CONTROL
    assert ble.writes[0][1][0] == pb.OP_EPOCH

    ble.subs[pb.CHAR_STATUS](_status(seq=1))
    ble.subs[pb.CHAR_STATUS](_status(seq=2, tick=2000))   # same snapshot
    ble.subs[pb.CHAR_STATUS](_status(seq=3, bl=80))
    statuses = [o for o in seen if o.schema == "wearable.status.v1"]
    assert len(statuses) == 2
    assert statuses[0].subject == "device:buds"
    assert statuses[1].value["battery_pct"] == 80

    ble.subs[pb.CHAR_EVENT](_event(seq=1, action=0x0001))   # stock action
    ble.subs[pb.CHAR_EVENT](_event(seq=4, action=0x0010))   # thoth, gap 2
    gestures = [o for o in seen if o.schema == "wearable.gesture.v1"]
    assert len(gestures) == 2
    assert link.dropped_events == 2
    assert len(events) == 1 and events[0][0] == "wearable_gesture"
    assert events[0][1]["action"] == "thoth_voice_toggle"

    assert link.set_session_state("speaking")
    assert ble.writes[-1] == (pb.CHAR_CONTROL, b"\x01\x03")
    st = link.status()
    assert st["connected"] and st["dropped_events"] == 2
