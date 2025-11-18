from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .preprocess import to_gray, binarize, dilate


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int

    def bbox(self) -> List[int]:
        return [self.x, self.y, self.w, self.h]


class DeltaGate:
    """Downscaled absdiff gate to decide if heavy OCR is needed.

    Keeps previous small frame; if change below threshold, skip.
    """

    def __init__(self, thr: int = 12):
        self.prev_small: np.ndarray | None = None
        self.thr = thr

    def changed(self, img_rgb: np.ndarray) -> bool:
        small = cv2.resize(img_rgb, (64, 36), interpolation=cv2.INTER_AREA)
        gray = to_gray(small)
        if self.prev_small is None:
            self.prev_small = gray
            return True
        diff = cv2.absdiff(self.prev_small, gray)
        score = float(np.mean(diff))
        self.prev_small = gray
        return score >= self.thr


def _merge_rects(rects: List[Tuple[int, int, int, int]], iou_thr: float = 0.15) -> List[ROI]:
    # Simple greedy merge by IoU
    def iou(a, b):
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    rects = rects[:]
    merged: List[Tuple[int, int, int, int]] = []
    while rects:
        a = rects.pop(0)
        changed = True
        while changed:
            changed = False
            keep = []
            for b in rects:
                if iou(a, b) >= iou_thr:
                    # merge
                    x1 = min(a[0], b[0])
                    y1 = min(a[1], b[1])
                    x2 = max(a[0] + a[2], b[0] + b[2])
                    y2 = max(a[1] + a[3], b[1] + b[3])
                    a = (x1, y1, x2 - x1, y2 - y1)
                    changed = True
                else:
                    keep.append(b)
            rects = keep
        merged.append(a)
    return [ROI(*m) for m in merged]


def extract_rois(img_rgb: np.ndarray, min_area: int = 120, max_rois: int = 12) -> List[ROI]:
    gray = to_gray(img_rgb)
    mask = binarize(gray)
    mask = dilate(mask, 3, 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            rects.append((x, y, w, h))
    rois = _merge_rects(sorted(rects, key=lambda r: r[0])[:max_rois])
    return rois


class ROICooldown:
    """Cooldown using dHash of cropped ROI to avoid jitter duplicates."""

    def __init__(self, cooldown_ms: int = 600):
        self.cooldown_ms = cooldown_ms
        self.last_seen: Dict[int, float] = {}

    @staticmethod
    def dhash(img_gray: np.ndarray, size: int = 8) -> int:
        img = cv2.resize(img_gray, (size + 1, size), interpolation=cv2.INTER_AREA)
        diff = img[:, 1:] > img[:, :-1]
        bits = 0
        for v in diff.flatten():
            bits = (bits << 1) | int(v)
        return bits

    def allow(self, patch_gray: np.ndarray) -> bool:
        key = self.dhash(patch_gray)
        now = monotonic() * 1000.0
        last = self.last_seen.get(key, -1e9)
        if (now - last) >= self.cooldown_ms:
            self.last_seen[key] = now
            return True
        return False

