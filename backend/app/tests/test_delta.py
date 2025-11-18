import numpy as np

from app.ocr.delta import DeltaGate, extract_rois


def test_delta_gate_change_vs_static():
    g = DeltaGate(thr=5)
    a = np.zeros((200, 300, 3), dtype=np.uint8)
    b = a.copy()
    b[50:80, 60:90, :] = 255
    assert g.changed(a) is True
    assert g.changed(a) is False  # no change
    assert g.changed(b) is True   # change detected


def test_extract_rois_and_cap():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[10:30, 10:80] = 255
    img[100:160, 120:200] = 255
    rois = extract_rois(img, min_area=50, max_rois=1)
    assert len(rois) == 1

