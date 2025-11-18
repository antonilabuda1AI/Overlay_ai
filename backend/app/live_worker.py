from __future__ import annotations

import threading
import time
from typing import Optional, List, Tuple

import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

from .config import SETTINGS
from .ocr.capture import BBox
from .ocr.delta import DeltaGate, ROICooldown, extract_rois
from .ocr.tesseract import batch_ocr
from .storage.session_db import SessionDB
from .ws import LIVE_HUB
from .perf import PerfLogger, TickStats


class LiveObserver:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = False
        self._monitor_id: Optional[int] = None
        self._bbox: Optional[BBox] = None
        self._session_db: Optional[SessionDB] = None
        self._gate = DeltaGate(thr=SETTINGS.delta.thr)
        self._cool = ROICooldown(cooldown_ms=SETTINGS.roi_cooldown_ms)
        self._ocr_pool: Optional[ThreadPoolExecutor] = None
        self._perf = PerfLogger()
        self._busy = threading.Event()  # OCR in progress flag for debounce

    def start(self, db: SessionDB, monitor_id: Optional[int], bbox: Optional[list[int]]):
        if self._running:
            return
        self._session_db = db
        self._monitor_id = monitor_id
        self._bbox = BBox(*bbox) if bbox else None
        self._stop.clear()
        self._ocr_pool = ThreadPoolExecutor(max_workers=max(1, SETTINGS.ocr_pool_size), thread_name_prefix="ocr")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._running = True

    def stop(self):
        if not self._running:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._ocr_pool:
            self._ocr_pool.shutdown(wait=False, cancel_futures=True)
        self._running = False

    def _run(self):
        interval_ms = min(max(SETTINGS.interval_ms, SETTINGS.interval_ms_min), SETTINGS.interval_ms_max)
        interval = max(20, interval_ms) / 1000.0
        try:
            from mss import mss
        except Exception:
            # mss not available; sleep-loop to avoid tight spin
            while not self._stop.is_set():
                time.sleep(interval)
            return
        with mss() as sct:
            # choose monitor region once
            if self._bbox is not None:
                mon = {"left": self._bbox.left, "top": self._bbox.top, "width": self._bbox.width, "height": self._bbox.height}
            else:
                mlist = sct.monitors
                idx = 1 if self._monitor_id is None else max(1, min(self._monitor_id, len(mlist) - 1))
                mon = mlist[idx]
            while not self._stop.is_set():
                tick_start = time.perf_counter_ns()
                s = TickStats(ts_ms=int(time.time() * 1000))
                try:
                    # capture
                    t_cap0 = time.perf_counter_ns()
                    raw = sct.grab(mon)
                    arr = np.array(raw)
                    img = arr[:, :, :3][:, :, ::-1]
                    s.capture_ms = (time.perf_counter_ns() - t_cap0) / 1e6

                    # delta gate
                    t_delta0 = time.perf_counter_ns()
                    if SETTINGS.delta.enabled and not self._gate.changed(img):
                        s.delta_ms = (time.perf_counter_ns() - t_delta0) / 1e6
                        s.total_ms = (time.perf_counter_ns() - tick_start) / 1e6
                        self._perf.log(s)
                        sleep_left = interval - s.total_ms / 1000.0
                        if sleep_left > 0:
                            time.sleep(sleep_left)
                        continue

                    rois = extract_rois(img, SETTINGS.delta.min_area, SETTINGS.delta.max_rois)
                    s.delta_ms = (time.perf_counter_ns() - t_delta0) / 1e6

                    if rois:
                        kept = []
                        budget_ms = 20.0
                        for r in rois[: SETTINGS.delta.max_rois_per_tick]:
                            crop = img[r.y : r.y + r.h, r.x : r.x + r.w]
                            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                            if self._cool.allow(gray):
                                kept.append(r)
                            if (time.perf_counter_ns() - t_delta0) / 1e6 > budget_ms:
                                break
                        if kept:
                            if self._busy.is_set():
                                s.dropped_frames += 1
                            else:
                                self._busy.set()
                                t_ocr0 = time.perf_counter_ns()
                                boxes = [r.bbox() for r in kept]
                                whitelist = SETTINGS.whitelist_regex

                                def do_ocr():
                                    return batch_ocr(img, boxes, SETTINGS.psm, SETTINGS.ocr_langs, whitelist=whitelist)

                                try:
                                    if self._ocr_pool and SETTINGS.ocr_pool_size > 1:
                                        texts = self._ocr_pool.submit(do_ocr).result()
                                    else:
                                        texts = do_ocr()
                                finally:
                                    self._busy.clear()
                                s.ocr_ms = (time.perf_counter_ns() - t_ocr0) / 1e6
                                now_ms = int(time.time() * 1000)
                                t_db0 = time.perf_counter_ns()
                                for r, txt in zip(kept, texts):
                                    if not txt.strip():
                                        s.skipped_rois += 1
                                        continue
                                    s.ocr_hits += 1
                                    if self._session_db is not None:
                                        self._session_db.append_event(now_ms, r.bbox(), txt, source="ocr")
                                    try:
                                        import anyio
                                        anyio.from_thread.run(
                                            LIVE_HUB.broadcast,
                                            {"ts_ms": now_ms, "type": "ocr", "text": txt, "bbox": r.bbox()},
                                        )
                                    except Exception:
                                        pass
                                s.db_ms = (time.perf_counter_ns() - t_db0) / 1e6
                                s.rois = len(boxes)
                except Exception:
                    pass
                s.total_ms = (time.perf_counter_ns() - tick_start) / 1e6
                self._perf.log(s)
                sleep_left = interval - s.total_ms / 1000.0
                if sleep_left > 0:
                    time.sleep(sleep_left)


LIVE_OBSERVER = LiveObserver()
