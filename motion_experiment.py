#!/usr/bin/env python3
"""Experiment: ROI pixel-change motion detection on a recorded video.

Covers pipeline stages 1-4 only (frame source, ROI preprocess, motion measure,
MotionGate). No recording, no MQTT. The goal is to look at the numbers on your
own clips and pick thresholds from data.

Usage:
    python motion_experiment.py clip.mp4                    # draw ROI in a window
    python motion_experiment.py clip.mp4 --roi 400,300,600,200
    python motion_experiment.py clip.mp4 --roi 400,300,600,200 --annotate

Outputs go to ./out/<clip name>_*  (csv, png, optional annotated mp4).

Rules kept from the repo: the detector and the gate are pure (no clock, no IO),
and every timestamp is the frame's own time, never "when we noticed".
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Stage 3: motion measure (pure)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    ts: float  # frame capture time, seconds
    changed_frac: float  # share of ROI pixels that changed vs the lagged frame
    shift_px_s: float  # coherent sideways shift, original pixels per second
    response: float  # phase-correlation confidence, 0..1 (low = noise)


class RoiMotion:
    """Compares the ROI now against the ROI about `lag_s` seconds earlier."""

    def __init__(
        self,
        roi: tuple[int, int, int, int],
        lag_s: float,
        diff_thresh: int,
        width: int,
    ) -> None:
        self._x, self._y, self._w, self._h = roi
        self._lag_s = lag_s
        self._diff_thresh = diff_thresh
        self._size = (width, max(int(round(self._h * width / self._w)), 8))
        self._scale = width / self._w
        self._window = cv2.createHanningWindow(self._size, cv2.CV_32F)
        self._kernel = np.ones((3, 3), np.uint8)
        self._hist: deque[tuple[float, np.ndarray]] = deque()

    def _prep(self, frame_bgr: np.ndarray) -> np.ndarray:
        crop = frame_bgr[self._y : self._y + self._h, self._x : self._x + self._w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self._size, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def update(self, ts: float, frame_bgr: np.ndarray) -> Sample | None:
        cur = self._prep(frame_bgr)
        self._hist.append((ts, cur))
        # keep the newest frame that is at least `lag_s` old as the reference
        while len(self._hist) >= 2 and self._hist[1][0] <= ts - self._lag_s:
            self._hist.popleft()
        ref_ts, ref = self._hist[0]
        dt = ts - ref_ts
        if dt < self._lag_s * 0.9:
            return None  # warm-up: not enough history yet

        diff = cv2.absdiff(cur, ref)
        binary = (diff > self._diff_thresh).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._kernel)
        frac = float(binary.mean())

        (dx, _dy), resp = cv2.phaseCorrelate(
            ref.astype(np.float32), cur.astype(np.float32), self._window
        )
        shift = dx / self._scale / dt if resp > 0.1 else 0.0
        return Sample(ts, frac, float(shift), float(resp))


# --------------------------------------------------------------------------
# Stage 4: MotionGate, hysteresis (pure)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateEvent:
    kind: str  # "MOVING" or "STILL"
    event_ts: float  # back-dated: when it physically happened (frame time)
    decided_ts: float  # when the gate became sure (frame time)


class MotionGate:
    """STILL -> MOVING after `confirm` consecutive samples above `on`.
    MOVING -> STILL after the value stays below `off` for `hold_s` seconds.
    Between off and on nothing changes (dead zone).

    MOVING is stamped with the first sample of the confirmed run.
    STILL is stamped with the last sample that was still at or above `off`.
    """

    def __init__(self, on: float, off: float, confirm: int, hold_s: float) -> None:
        if off >= on:
            raise ValueError("off threshold must be lower than on threshold")
        self._on, self._off = on, off
        self._confirm, self._hold_s = confirm, hold_s
        self.state = "STILL"
        self._run_start: float | None = None
        self._run_n = 0
        self._last_alive: float | None = None
        self._below_since: float | None = None

    def update(self, s: Sample) -> GateEvent | None:
        if self.state == "STILL":
            if s.changed_frac > self._on:
                if self._run_n == 0:
                    self._run_start = s.ts
                self._run_n += 1
                if self._run_n >= self._confirm:
                    start = self._run_start if self._run_start is not None else s.ts
                    self.state = "MOVING"
                    self._run_n, self._run_start = 0, None
                    self._last_alive, self._below_since = s.ts, None
                    return GateEvent("MOVING", start, s.ts)
            else:
                self._run_n, self._run_start = 0, None
            return None

        # state == MOVING
        if s.changed_frac >= self._off:
            self._last_alive, self._below_since = s.ts, None
            return None
        if self._below_since is None:
            self._below_since = s.ts
        if s.ts - self._below_since >= self._hold_s:
            self.state = "STILL"
            end = self._last_alive if self._last_alive is not None else s.ts
            self._last_alive, self._below_since = None, None
            return GateEvent("STILL", end, s.ts)
        return None


# --------------------------------------------------------------------------
# Stage 1: frame source (video file). Swap for a live source later.
# --------------------------------------------------------------------------


def read_frames(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx / fps, frame  # frame time from index; fine for file replay
            idx += 1
    finally:
        cap.release()


def pick_roi(path: str, frame_idx: int, view_width: int) -> tuple[int, int, int, int]:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read a frame to draw the ROI on")
    print("Click and drag a box over the train path, then press ENTER or SPACE to confirm.")
    print("Press C to cancel. Use --roi-frame N to pick a frame where the train is visible.")
    # shrink big frames so the whole picture fits on screen
    scale = min(1.0, view_width / frame.shape[1], view_width * 0.6 / frame.shape[0])
    shown = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
    sx, sy, sw, sh = cv2.selectROI("draw ROI", shown, showCrosshair=False)
    cv2.destroyAllWindows()
    if sw == 0 or sh == 0:
        raise SystemExit("no ROI selected")
    # map back to original pixel coordinates
    x, y, w, h = (int(round(v / scale)) for v in (sx, sy, sw, sh))
    print(f"ROI: --roi {x},{y},{w},{h}   (reuse this to skip the window)")
    return int(x), int(y), int(w), int(h)


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


def save_plot(path: Path, samples: list[Sample], events: list[GateEvent], args) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot (pip install matplotlib)")
        return

    ts = [s.ts for s in samples]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.plot(ts, [s.changed_frac for s in samples], lw=1.2, label="changed_frac")
    a1.axhline(args.on, color="g", ls="--", lw=0.8, label=f"ON {args.on}")
    a1.axhline(args.off, color="r", ls="--", lw=0.8, label=f"OFF {args.off}")
    a1.set_ylabel("changed fraction")
    a1.legend(loc="upper right")

    a2.plot(ts, [s.shift_px_s for s in samples], lw=1.0, color="tab:orange")
    a2.set_ylabel("shift px/s (x)")
    a2.set_xlabel("time (s)")

    start = None
    for e in events:
        for ax in (a1, a2):
            ax.axvline(e.event_ts, color="g" if e.kind == "MOVING" else "r", lw=1.0)
        if e.kind == "MOVING":
            start = e.event_ts
        elif start is not None:
            for ax in (a1, a2):
                ax.axvspan(start, e.event_ts, color="g", alpha=0.12)
            start = None
    if start is not None:  # still moving at end of clip
        for ax in (a1, a2):
            ax.axvspan(start, ts[-1], color="g", alpha=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def save_grid_frame(video: str, frame_idx: int, out_dir: Path) -> None:
    """Save one frame with a labelled pixel grid. Read the ROI off it, or share it."""
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read that frame")
    h, w = frame.shape[:2]
    step = 100 if max(w, h) > 600 else 50
    img = frame.copy()
    col = (0, 255, 255)
    for gx in range(0, w, step):
        cv2.line(img, (gx, 0), (gx, h - 1), col, 1)
        cv2.putText(img, str(gx), (gx + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    for gy in range(0, h, step):
        cv2.line(img, (0, gy), (w - 1, gy), col, 1)
        cv2.putText(img, str(gy), (3, gy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    out = out_dir / f"{Path(video).stem}_frame{frame_idx}_grid.png"
    cv2.imwrite(str(out), img)
    print(f"Video is {w}x{h}. Saved {out}")
    print("Open it in any image viewer. ROI = x,y,w,h where x,y is the top-left corner.")


def suggest_roi(video: str, out_dir: Path, lag_s: float = 0.4, rate_hz: float = 10.0) -> None:
    """Find where pixels change over the whole clip and propose a box around it."""
    small: list[np.ndarray] = []
    first = None
    last_ts = -1e9
    scale = 1.0
    for ts, frame in read_frames(video):
        if first is None:
            first = frame
            scale = 320 / frame.shape[1]
        if ts - last_ts >= 1.0 / rate_hz - 1e-6:
            last_ts = ts
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            small.append(cv2.GaussianBlur(g, (5, 5), 0))
    k = max(1, round(lag_s * rate_hz))
    if first is None or len(small) <= k:
        raise SystemExit("clip too short for --suggest-roi")
    acc = np.zeros(small[0].shape, np.float32)
    for i in range(k, len(small)):
        acc += (cv2.absdiff(small[i], small[i - k]) > 18).astype(np.float32)
    heat = acc / (len(small) - k)
    if heat.max() < 0.05:
        print("Almost nothing moves in this clip, so there is nothing to suggest.")
        return
    mask = (heat >= 0.3 * heat.max()).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bx, by, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    fh, fw = first.shape[:2]
    pad = int(0.03 * fw)
    x = max(int(bx / scale) - pad, 0)
    y = max(int(by / scale) - pad, 0)
    x2 = min(int((bx + bw) / scale) + pad, fw)
    y2 = min(int((by + bh) / scale) + pad, fh)
    heat_img = cv2.applyColorMap((255 * heat / heat.max()).astype(np.uint8), cv2.COLORMAP_JET)
    heat_img = cv2.resize(heat_img, (fw, fh))
    vis = cv2.addWeighted(first, 0.55, heat_img, 0.45, 0)
    cv2.rectangle(vis, (x, y), (x2, y2), (255, 255, 255), 3)
    out = out_dir / f"{Path(video).stem}_suggested_roi.png"
    cv2.imwrite(str(out), vis)
    print(f"Suggested ROI: --roi {x},{y},{x2 - x},{y2 - y}")
    print(f"Saved {out}  (red = most change, white box = suggestion). Check it before using.")
    print("If people or other things also move in this clip, the box will include them.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--roi", help="x,y,w,h in original pixels; omit to draw it in a window")
    ap.add_argument("--roi-frame", type=int, default=0, help="frame to draw the ROI on")
    ap.add_argument("--view-width", type=int, default=800, help="ROI window width in screen pixels")
    ap.add_argument("--grid-frame", action="store_true", help="save a frame with a pixel grid, then exit")
    ap.add_argument("--suggest-roi", action="store_true", help="propose an ROI from motion in the clip, then exit")
    ap.add_argument("--sample-hz", type=float, default=10.0, help="analysis rate")
    ap.add_argument("--lag", type=float, default=0.8, help="seconds back to compare against")
    ap.add_argument("--diff-thresh", type=int, default=18, help="gray-level change that counts")
    ap.add_argument("--width", type=int, default=320, help="ROI is shrunk to this width")
    ap.add_argument("--on", type=float, default=0.04)
    ap.add_argument("--off", type=float, default=0.01)
    ap.add_argument("--confirm", type=int, default=3, help="samples in a row above ON")
    ap.add_argument("--hold", type=float, default=2.0, help="seconds below OFF to call STILL")
    ap.add_argument("--annotate", action="store_true", help="write an annotated mp4")
    ap.add_argument("--out", default="out", help="output folder")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.grid_frame:
        save_grid_frame(args.video, args.roi_frame, Path(args.out))
        return
    if args.suggest_roi:
        suggest_roi(args.video, Path(args.out))
        return

    roi = (
        tuple(int(v) for v in args.roi.split(","))
        if args.roi
        else pick_roi(args.video, args.roi_frame, args.view_width)
    )
    if len(roi) != 4:
        raise SystemExit("--roi needs x,y,w,h")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / Path(args.video).stem

    detector = RoiMotion(roi, args.lag, args.diff_thresh, args.width)
    gate = MotionGate(args.on, args.off, args.confirm, args.hold)

    samples: list[Sample] = []
    events: list[GateEvent] = []
    writer = None
    last_ts = -1e9
    shown_frac, shown_state = 0.0, gate.state
    min_gap = 1.0 / args.sample_hz - 1e-6
    n_frames = 0
    x, y, w, h = roi

    for ts, frame in read_frames(args.video):
        n_frames += 1
        if ts - last_ts >= min_gap:
            last_ts = ts
            s = detector.update(ts, frame)
            if s is not None:
                samples.append(s)
                ev = gate.update(s)
                if ev is not None:
                    events.append(ev)
                    print(
                        f"{ev.kind:6s} event_ts={ev.event_ts:7.2f}s  "
                        f"decided_at={ev.decided_ts:7.2f}s"
                    )
                shown_frac, shown_state = s.changed_frac, gate.state

        if args.annotate:
            vis = frame.copy()
            color = (0, 200, 0) if shown_state == "MOVING" else (0, 0, 220)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                vis, f"{shown_state}  frac={shown_frac:.3f}  t={ts:.2f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(stem) + "_annotated.mp4",
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    25.0,
                    (vis.shape[1], vis.shape[0]),
                )
            writer.write(vis)

    if writer is not None:
        writer.release()

    if not samples:
        print(
            f"No samples produced from {n_frames} frames. The clip is probably shorter "
            f"than the lag ({args.lag}s). Use a longer clip or a smaller --lag."
        )
        return

    if gate.state == "MOVING":
        print("Clip ended while still MOVING (no STILL event, which is expected for a short clip).")

    with open(str(stem) + "_samples.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["ts", "changed_frac", "shift_px_s", "response"])
        for s in samples:
            wr.writerow([f"{s.ts:.3f}", f"{s.changed_frac:.5f}", f"{s.shift_px_s:.2f}", f"{s.response:.3f}"])

    save_plot(Path(str(stem) + "_plot.png"), samples, events, args)

    fr = np.array([s.changed_frac for s in samples])
    print(
        f"\n{n_frames} frames, {len(samples)} samples. changed_frac: "
        f"min={fr.min():.4f} median={np.median(fr):.4f} p95={np.percentile(fr, 95):.4f} max={fr.max():.4f}"
    )
    print(f"Files in {out_dir}/: *_samples.csv, *_plot.png" + (", *_annotated.mp4" if args.annotate else ""))


if __name__ == "__main__":
    main()