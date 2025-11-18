from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns
from typing import Optional

from .config import SETTINGS


@dataclass
class TickStats:
    ts_ms: int
    capture_ms: float = 0.0
    delta_ms: float = 0.0
    ocr_ms: float = 0.0
    db_ms: float = 0.0
    total_ms: float = 0.0
    rois: int = 0
    skipped_rois: int = 0
    ocr_hits: int = 0
    cache_hits: int = 0
    dropped_frames: int = 0


class PerfLogger:
    def __init__(self) -> None:
        log_dir = SETTINGS.data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d")
        self.path = log_dir / f"perf-{stamp}.csv"
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "ts_ms","capture_ms","delta_ms","ocr_ms","db_ms","total_ms",
                    "rois","skipped_rois","ocr_hits","cache_hits","dropped_frames"
                ])

    def log(self, s: TickStats) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                s.ts_ms, f"{s.capture_ms:.3f}", f"{s.delta_ms:.3f}", f"{s.ocr_ms:.3f}",
                f"{s.db_ms:.3f}", f"{s.total_ms:.3f}", s.rois, s.skipped_rois, s.ocr_hits,
                s.cache_hits, s.dropped_frames
            ])

    @staticmethod
    def now_ms() -> int:
        return int(perf_counter_ns() / 1_000_000)

