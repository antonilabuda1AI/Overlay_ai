from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def binarize(gray: np.ndarray) -> np.ndarray:
    # Adaptive threshold is robust to lighting
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 11)


_KERNELS: dict[tuple[int, int], np.ndarray] = {}


def get_kernel(ksize: int = 3) -> np.ndarray:
    key = (ksize, ksize)
    k = _KERNELS.get(key)
    if k is None:
        k = np.ones((ksize, ksize), np.uint8)
        _KERNELS[key] = k
    return k


def dilate(mask: np.ndarray, ksize: int = 3, iters: int = 1) -> np.ndarray:
    return cv2.dilate(mask, get_kernel(ksize), iterations=iters)
