from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from mss import mss


@dataclass
class BBox:
    left: int
    top: int
    width: int
    height: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)


def grab_frame(monitor_id: Optional[int] = None, bbox: Optional[BBox] = None) -> np.ndarray:
    """Grab a screen frame as numpy array in RGB.

    If bbox is provided, capture region; otherwise capture the entire primary monitor or monitor_id.
    """
    with mss() as sct:
        if bbox is not None:
            mon = {"left": bbox.left, "top": bbox.top, "width": bbox.width, "height": bbox.height}
        else:
            if monitor_id is None:
                mon = sct.monitors[1]
            else:
                # mss uses 1-based index for monitors
                mon = sct.monitors[max(1, min(monitor_id, len(sct.monitors) - 1))]
        img = sct.grab(mon)
        # BGRA -> RGB
        arr = np.array(img)
        return arr[:, :, :3][:, :, ::-1]

