"""Capture PineTime/InfiniTime IMU via the thoth/whispy SDK.

Connects centrally over BLE, subscribes the motion characteristic
(stamped ``00030003`` preferred, legacy ``00030002`` polled at 10 Hz
otherwise — same fallback as WatchLink), folds samples through
``thoth.bluetooth.pinetime.ImuWindowBuffer``, and writes 2 s
``imu.window.v1`` windows as whispy ``Observation`` dicts.

Usage:
    python watch_imu_capture.py [--address FF:96:7E:5A:69:41]
        [--duration 300] [--out-dir "%USERPROFILE%\\.thoth\\captures"]

Outputs (in --out-dir):
    imu_raw_<ts>.csv        t_s,x_g,y_g,z_g[,tick_ms,seq]
    imu_windows_<ts>.jsonl  one imu.window.v1 Observation per line

The watch is a single-central device — disconnect it from the phone
app (or stop the node daemon) before running, or connect() will hang
on a device that isn't advertising.
"""
import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thoth.bluetooth.pinetime import (  # noqa: E402
    CHAR_MOTION_STAMPED, CHAR_RAW_MOTION, MOTION_SERVICE,
    ImuWindowBuffer, parse_motion, parse_motion_stamped)
from whispy.contracts import Observation  # noqa: E402

WATCH_ID = "device:81a5af81-1151-409f-94e3-75883bf352e1"
WATCH_UUID = "42d6921a-9872-5a5d-a9da-bacc6f908680"  # uuid5(dns, ble id)
SRC = "pinetime:bma421"
BRAIN = "https://api.thothcraft.com"


def _find_watch(address: str | None):
    from bleak import BleakScanner

    async def _scan():
        if address:
            dev = await BleakScanner.find_device_by_address(
                address, timeout=10.0)
            if dev:
                return dev
        def _like(d, adv):
            name = (adv.local_name or d.name or "").lower()
            return "infinitime" in name or "pinetime" in name
        return await BleakScanner.find_device_by_filter(_like,
                                                        timeout=15.0)
    return asyncio.run(_scan())


async def capture(dev, seconds: float, out_dir: Path) -> None:
    from bleak import BleakClient

    raw_path = out_dir / f"imu_raw_{int(time.time())}.csv"
    win_path = out_dir / f"imu_windows_{int(time.time())}.csv"
    win_path = win_path.with_suffix(".jsonl")

    buf = ImuWindowBuffer()
    buf.dropped = 0
    t0 = time.time()
    n_raw = 0
    n_win = 0
    stamped = False

    csv_f = raw_path.open("w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["t_s", "x_g", "y_g", "z_g", "tick_ms", "seq"])
    jf = win_path.open("w", encoding="utf-8")

    def flush_window(now: float) -> None:
        nonlocal n_win
        if not buf.ready(now):
            return
        value = buf.flush()
        if not value:
            return
        obs = Observation(
            source_id=SRC, device_id=WATCH_ID,
            timestamp=now, payload={"value": value},
            schema="imu.window.v1",
            provenance={"collector": "watch_imu_capture",
                        "subject": WATCH_ID})
        jf.write(json.dumps(obs.to_dict()) + "\n")
        jf.flush()
        n_win += 1

    async with BleakClient(dev, timeout=15.0) as client:
        svcs = {c.uuid for s in client.services for c in s.characteristics}
        stamped = CHAR_MOTION_STAMPED in svcs
        legacy = CHAR_RAW_MOTION in svcs
        print(f"[cap] services: stamped={stamped} legacy={legacy}")
        if not (stamped or legacy):
            raise RuntimeError("no motion characteristic on this watch")

        def on_motion(_, data: bytes) -> None:
            nonlocal n_raw
            now = time.time()
            if stamped:
                p = parse_motion_stamped(bytes(data))
                if not p:
                    return
                x, y, z, tick_ms, seq = p
            else:
                p = parse_motion(bytes(data))
                if not p:
                    return
                x, y, z = p
                tick_ms = seq = None
            n_raw += 1
            buf.add(x, y, z, now, tick_ms)
            csv_w.writerow([round(now - t0, 3), round(x, 5),
                            round(y, 5), round(z, 5),
                            tick_ms or "", seq or ""])
            flush_window(now)

        char = CHAR_MOTION_STAMPED if stamped else CHAR_RAW_MOTION
        await client.start_notify(char, on_motion)
        print(f"[cap] notify on {char[-4:]} — recording {seconds:.0f}s …")

        # Legacy char only notifies on change — poll reads at 10 Hz so a
        # still watch still yields samples (same trick as the app).
        async def poll():
            while True:
                await asyncio.sleep(0.1)
                try:
                    data = await client.read_gatt_char(CHAR_RAW_MOTION)
                    on_motion(None, data)
                except Exception:
                    pass

        poll_task = (asyncio.create_task(poll())
                     if not stamped else None)
        end = t0 + seconds
        try:
            while time.time() < end:
                await asyncio.sleep(0.5)
                now = time.time()
                flush_window(now)
                if int(now - t0) % 30 == 0:
                    print(f"[cap] t={int(now - t0):>3}s "
                          f"samples={n_raw} windows={n_win}",
                          flush=True)
        finally:
            if poll_task:
                poll_task.cancel()
            await client.stop_notify(char)

    flush_window(time.time() + 1e9)
    csv_f.close()
    jf.close()
    print(f"[cap] done — {n_raw} samples, {n_win} windows")
    print(f"      raw:     {raw_path}")
    print(f"      windows: {win_path}")


def capture_from_brain(device_uuid: str, api_key: str, seconds: float,
                       out_dir: Path) -> None:
    """Pull the phone-relayed IMU stream (watch → phone → Brain live
    chunks) and window it through the same whispy contract — the path
    to use while the phone holds the BLE link."""
    import urllib.request

    url = (f"{BRAIN}/v1/devices/{device_uuid}/streams/pinetime-motion")
    raw_path = out_dir / f"imu_raw_{int(time.time())}.csv"
    win_path = out_dir / f"imu_windows_{int(time.time())}.jsonl"
    buf = ImuWindowBuffer()
    t0 = time.time()
    n_raw = n_win = 0
    seen = set()           # chunks re-deliver earlier rows — dedup
    csv_f = raw_path.open("w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["t_s", "x_g", "y_g", "z_g", "tick_ms", "seq"])
    jf = win_path.open("w", encoding="utf-8")
    cursor = None
    while time.time() < t0 + seconds:
        req_url = url + (f"?cursor={cursor}" if cursor else "")
        req = urllib.request.Request(
            req_url, headers={"X-Api-Key": api_key})
        try:
            page = json.loads(urllib.request.urlopen(req, timeout=20)
                              .read())
        except Exception as e:
            print(f"[brain] poll error: {e}")
            time.sleep(8)
            continue
        cursor = page.get("cursor") or cursor
        for s in page.get("samples") or []:
            p = s.get("payload") or {}
            key = (p.get("tick_ms"), s.get("sequence"))
            if key in seen:
                continue
            seen.add(key)
            ts = float(s.get("timestamp") or time.time())
            x, y, z = p.get("acc_x"), p.get("acc_y"), p.get("acc_z")
            if x is None or y is None or z is None:
                continue
            n_raw += 1
            buf.add(float(x), float(y), float(z), ts,
                    p.get("tick_ms"))
            csv_w.writerow([round(ts - t0, 3), x, y, z,
                            p.get("tick_ms") or "",
                            s.get("sequence") or ""])
            if buf.ready(ts):
                value = buf.flush()
                if value:
                    obs = Observation(
                        source_id=SRC, device_id=WATCH_ID,
                        timestamp=ts, payload={"value": value},
                        schema="imu.window.v1",
                        provenance={"collector": "brain-stream",
                                    "subject": WATCH_ID})
                    jf.write(json.dumps(obs.to_dict()) + "\n")
                    jf.flush()
                    n_win += 1
        csv_f.flush()
        print(f"[brain] t={int(time.time() - t0):>3}s "
              f"samples={n_raw} windows={n_win} "
              f"state={page.get('state')}", flush=True)
        time.sleep(8)
    value = buf.flush()
    if value:
        obs = Observation(source_id=SRC, device_id=WATCH_ID,
                          timestamp=time.time(),
                          payload={"value": value},
                          schema="imu.window.v1")
        jf.write(json.dumps(obs.to_dict()) + "\n")
        n_win += 1
    csv_f.close()
    jf.close()
    print(f"[brain] done — {n_raw} samples, {n_win} windows")
    print(f"        raw:     {raw_path}")
    print(f"        windows: {win_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default="FF:96:7E:5A:69:41")
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--out-dir",
                    default=str(Path.home() / ".thoth" / "captures"))
    ap.add_argument("--from-brain", action="store_true",
                    help="pull the phone-relayed stream instead of a "
                         "direct BLE link (watch can stay on the phone)")
    ap.add_argument("--api-key", default=os.environ.get("THOTH_API_KEY"))
    ap.add_argument("--device-uuid", default=WATCH_UUID)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_brain:
        if not args.api_key:
            print("[cap] --from-brain needs --api-key or THOTH_API_KEY")
            sys.exit(2)
        capture_from_brain(args.device_uuid, args.api_key,
                           args.duration, out_dir)
        return

    print(f"[cap] scanning for {args.address} …")
    dev = _find_watch(args.address)
    if dev is None:
        print("[cap] watch not advertising — it's likely still held by "
              "the phone or the node daemon. Disconnect it there first "
              "(BLE is single-central), then re-run. Or use "
              "--from-brain to capture the phone-relayed stream.")
        sys.exit(2)
    print(f"[cap] found {dev.address} ({dev.name})")
    asyncio.run(capture(dev, args.duration, out_dir))


if __name__ == "__main__":
    main()
