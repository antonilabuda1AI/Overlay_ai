#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from backend.app.config import SETTINGS
from backend.app.ocr.capture import BBox, grab_frame
from backend.app.ocr.delta import DeltaGate, ROICooldown, extract_rois
from backend.app.ocr.tesseract import batch_ocr
from backend.app.perf import PerfLogger, TickStats


def synthetic_frame(w: int = 640, h: int = 360, step: int = 0) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    x0 = (step * 7) % (w - 120)
    y0 = (step * 5) % (h - 40)
    img[y0:y0+24, x0:x0+120] = 255
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--interval-ms", type=int, default=200)
    ap.add_argument("--max-rois", type=int, default=5)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    perf = PerfLogger()
    gate = DeltaGate(thr=SETTINGS.delta.thr)
    cool = ROICooldown(cooldown_ms=SETTINGS.roi_cooldown_ms)

    for i in range(args.ticks):
        t_start = time.perf_counter_ns()
        s = TickStats(ts_ms=int(time.time() * 1000))
        if args.synthetic:
            t0 = time.perf_counter_ns()
            img = synthetic_frame(step=i)
            s.capture_ms = (time.perf_counter_ns() - t0) / 1e6
        else:
            t0 = time.perf_counter_ns()
            img = grab_frame(None, None)
            s.capture_ms = (time.perf_counter_ns() - t0) / 1e6
        t1 = time.perf_counter_ns()
        changed = gate.changed(img)
        if not changed:
            s.delta_ms = (time.perf_counter_ns() - t1) / 1e6
            s.total_ms = (time.perf_counter_ns() - t_start) / 1e6
        else:
            rois = extract_rois(img, SETTINGS.delta.min_area, min(SETTINGS.delta.max_rois, args.max_rois))
            s.delta_ms = (time.perf_counter_ns() - t1) / 1e6
            if rois:
                t2 = time.perf_counter_ns()
                boxes = []
                for r in rois[: SETTINGS.delta.max_rois_per_tick]:
                    crop = img[r.y : r.y + r.h, r.x : r.x + r.w]
                    gray = np.mean(crop, axis=2).astype("uint8")
                    if cool.allow(gray):
                        boxes.append(r.bbox())
                if boxes:
                    _ = batch_ocr(img, boxes, SETTINGS.psm, SETTINGS.ocr_langs)
                s.ocr_ms = (time.perf_counter_ns() - t2) / 1e6
                s.rois = len(boxes)
            s.total_ms = (time.perf_counter_ns() - t_start) / 1e6
        perf.log(s)
        print(f"{s.ts_ms},{s.capture_ms:.3f},{s.delta_ms:.3f},{s.ocr_ms:.3f},{s.db_ms:.3f},{s.total_ms:.3f},{s.rois},{s.skipped_rois},{s.ocr_hits},{s.cache_hits},{s.dropped_frames}")
        if s.total_ms < args.interval_ms:
            time.sleep((args.interval_ms - s.total_ms) / 1000.0)


if __name__ == "__main__":
    main()

