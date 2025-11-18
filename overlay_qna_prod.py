# -*- coding: utf-8 -*-
"""
Overlay Q&A - Top bar with Live OCR and RAG (FAISS + OpenAI)

Features
- Live OCR of the screen with caching and delta gating (simplified: always OCR per tick for reliability)
- Draggable answer bubble under the top bar
- History window with manual Save button
- Threads keep UI responsive (PyQt + QThread)
- Soft reset on window change (clears OCR/RAG context only)

Shortcuts
- Ctrl+L: Live OCR ON/OFF
- Ctrl+H: Toggle History
- Ctrl+R: Reset context (manual)
"""

import os, sys, time, ctypes, hashlib, datetime, re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image, ImageOps, ImageDraw
from mss import mss
import pytesseract
try:
    import faiss  # type: ignore
except Exception:  # allow running without faiss; fallback to NumPy search
    faiss = None
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt, QRect, QPoint, QTimer

# ================== CONFIG ==================
TESSERACT_CMD = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_CHAT  = "gpt-4o-mini"
OPENAI_MODEL_EMB   = "text-embedding-3-small"  # 1536 dim

TOPBAR_WIDTH       = 900
TOPBAR_HEIGHT      = 56

# Default OCR to pol+eng as discussed; change in Settings if needed
OCR_LANGS          = "pol+eng"
TESSERACT_CONFIG   = "--oem 3 --psm 6"

LIVE_OCR_INTERVAL_SEC     = 1.0     # base OCR interval
ECO_MAX_INTERVAL_SEC      = 6.0
ECO_STEP_SEC              = 1.0
ECO_STABLE_TICKS_TO_STEP  = 2

AUTO_OCR_MIN_PIXELS       = 200 * 100
DHASH_SIZE                = 8
DHASH_HAMMING_THRESHOLD   = 2
FORCE_OCR_TICKS           = 1       # reserved; we OCR each tick now

CONTEXT_TOP_K      = 4
CONTEXT_MAX_DOCS   = 24

ANSWER_AUTOHIDE_MS = 0   # do not auto-hide

OCR_WHITELIST      = os.getenv("OCR_WHITELIST", "")  # e.g. "A-Za-z0-9-_.:/"
PSM_AUTOTUNE       = True  # use psm=7 for very narrow/wide areas
PERF_LOG_CSV       = False

# Capture region: 'full' | 'center' | 'custom'
CAPTURE_MODE       = 'full'
CAPTURE_CUSTOM_BBOX = {"left":0,"top":0,"width":0,"height":0}

# Exclude the top bar area from screen capture to avoid self-OCR
CAPTURE_EXCLUDE_BAR = False
CAPTURE_EXCLUDE_MARGIN = 12  # pixels

# Toasts: popup only (tray fallback disabled)
TOAST_TRAY_FALLBACK = False  # tray fallback disabled
# ============================================


# ---------- Settings persistence ----------
def _settings_path():
    home = os.path.expanduser("~")
    d = os.path.join(home, ".studyglance")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "overlay_settings.json")


def load_user_settings():
    global OCR_LANGS, LIVE_OCR_INTERVAL_SEC, ECO_MAX_INTERVAL_SEC, ECO_STEP_SEC, ECO_STABLE_TICKS_TO_STEP
    global AUTO_OCR_MIN_PIXELS, DHASH_HAMMING_THRESHOLD, OCR_WHITELIST, PSM_AUTOTUNE, CAPTURE_MODE, CAPTURE_CUSTOM_BBOX
    try:
        import json
        p = _settings_path()
        if not os.path.exists(p):
            return
        with open(p, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        OCR_LANGS = cfg.get("OCR_LANGS", OCR_LANGS)
        LIVE_OCR_INTERVAL_SEC = float(cfg.get("LIVE_OCR_INTERVAL_SEC", LIVE_OCR_INTERVAL_SEC))
        ECO_MAX_INTERVAL_SEC = float(cfg.get("ECO_MAX_INTERVAL_SEC", ECO_MAX_INTERVAL_SEC))
        ECO_STEP_SEC = float(cfg.get("ECO_STEP_SEC", ECO_STEP_SEC))
        ECO_STABLE_TICKS_TO_STEP = int(cfg.get("ECO_STABLE_TICKS_TO_STEP", ECO_STABLE_TICKS_TO_STEP))
        AUTO_OCR_MIN_PIXELS = int(cfg.get("AUTO_OCR_MIN_PIXELS", AUTO_OCR_MIN_PIXELS))
        DHASH_HAMMING_THRESHOLD = int(cfg.get("DHASH_HAMMING_THRESHOLD", DHASH_HAMMING_THRESHOLD))
        OCR_WHITELIST = cfg.get("OCR_WHITELIST", OCR_WHITELIST)
        PSM_AUTOTUNE  = bool(cfg.get("PSM_AUTOTUNE", PSM_AUTOTUNE))
        CAPTURE_MODE = cfg.get("CAPTURE_MODE", CAPTURE_MODE)
        CAPTURE_CUSTOM_BBOX = cfg.get("CAPTURE_CUSTOM_BBOX", CAPTURE_CUSTOM_BBOX)
    except Exception:
        pass


def save_user_settings():
    try:
        import json
        cfg = {
            "OCR_LANGS": OCR_LANGS,
            "LIVE_OCR_INTERVAL_SEC": LIVE_OCR_INTERVAL_SEC,
            "ECO_MAX_INTERVAL_SEC": ECO_MAX_INTERVAL_SEC,
            "ECO_STEP_SEC": ECO_STEP_SEC,
            "ECO_STABLE_TICKS_TO_STEP": ECO_STABLE_TICKS_TO_STEP,
            "AUTO_OCR_MIN_PIXELS": AUTO_OCR_MIN_PIXELS,
            "DHASH_HAMMING_THRESHOLD": DHASH_HAMMING_THRESHOLD,
            "OCR_WHITELIST": OCR_WHITELIST,
            "PSM_AUTOTUNE": PSM_AUTOTUNE,
            "CAPTURE_MODE": CAPTURE_MODE,
            "CAPTURE_CUSTOM_BBOX": CAPTURE_CUSTOM_BBOX,
        }
        with open(_settings_path(), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
# ==========================================


# Set tesseract path if present
if TESSERACT_CMD and os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def tesseract_ok() -> bool:
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False



# ---------- OCR & screen helpers ----------
def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    # Fast binarization for Tesseract
    g = ImageOps.grayscale(img)
    # Fixed threshold works for most slides/windows
    g = g.point(lambda x: 255 if x > 170 else 0)
    return g


def _choose_psm(w: int, h: int, default_psm: int = 6) -> int:
    if not PSM_AUTOTUNE:
        return default_psm
    aspect = w / max(1, h)
    if w < 60 or h < 20 or aspect > 6.0:
        return 7
    return default_psm


def _filter_ocr_lines(lines: list[str]) -> list[str]:
    # UI/menu words and overlay controls
    # (keep "Answer" allowed now, as slides might contain it)
    UI_IGNORE_REGEX = re.compile(r"(live\s*ocr|eco|history|send|overlay|studyglance|topbar|ctrl\+l|ctrl\+h|close)", re.I)
    PATH_REGEX = re.compile(r"([A-Za-z]:\\|/home/|/Users/|\\overlay_ai\\|/overlay_ai/)")
    FILE_EXT_REGEX = re.compile(r"\b\w+\.(md|txt|py|json|toml|yaml|yml|exe|dll|venv|spec|dist|png|jpg)\b", re.I)
    out: list[str] = []
    for l in lines:
        s = l.strip()
        if not s:
            continue
        lo = s.lower()
        if lo.startswith("slide"):
            continue
        if UI_IGNORE_REGEX.search(s):
            continue
        if PATH_REGEX.search(s) or FILE_EXT_REGEX.search(s):
            continue
        # Heuristic: drop lines with very low alpha ratio (likely UI noise)
        letters = sum(ch.isalpha() for ch in s)
        ratio = letters / max(1, len(s))
        if ratio < 0.25 and len(s) >= 6:
            continue
        out.append(s)
    return out


def ocr_image(pil_img: Image.Image) -> str:
    img = preprocess_for_ocr(pil_img)
    psm = _choose_psm(img.width, img.height, 6)
    cfg = f"--oem 3 --psm {psm}"
    if OCR_WHITELIST:
        cfg += f" -c tessedit_char_whitelist={OCR_WHITELIST}"
    try:
        txt = pytesseract.image_to_string(img, config=cfg, lang=OCR_LANGS)
    except pytesseract.TesseractNotFoundError:
        txt = pytesseract.image_to_string(img, config=cfg)  # try with default langs
    lines = [l for l in txt.splitlines()]
    lines = _filter_ocr_lines(lines)
    return "\n".join(lines).strip()


def screen_for_widget(widget: QtWidgets.QWidget):
    return widget.screen() or QtGui.QGuiApplication.primaryScreen()


def dpi_scale_for_screen(scr: QtGui.QScreen):
    return (scr.logicalDotsPerInch() / 96.0) if scr else 1.0


def to_physical(rect: QRect, scale: float) -> dict:
    return {
        "left":   int(rect.left()   * scale),
        "top":    int(rect.top()    * scale),
        "width":  int(rect.width()  * scale),
        "height": int(rect.height() * scale),
    }


def capture_screen_bbox_of_widget(widget: QtWidgets.QWidget) -> dict:
    """Return bbox according to CAPTURE_MODE for the screen containing the bar.
       In full mode, may exclude the top bar if CAPTURE_EXCLUDE_BAR is enabled."""
    scr = screen_for_widget(widget)
    geo = scr.geometry()
    scale = dpi_scale_for_screen(scr)

    if CAPTURE_MODE == 'center':
        w = int(geo.width() * 0.6)
        h = int(geo.height() * 0.6)
        x = geo.left() + (geo.width() - w)//2
        y = geo.top() + (geo.height() - h)//2
        return to_physical(QRect(x, y, w, h), scale)

    if CAPTURE_MODE == 'custom' and CAPTURE_CUSTOM_BBOX.get("width",0)>0 and CAPTURE_CUSTOM_BBOX.get("height",0)>0:
        # Already in physical coords
        return dict(CAPTURE_CUSTOM_BBOX)

    # CAPTURE_MODE == 'full'
    if CAPTURE_EXCLUDE_BAR:
        try:
            bar_geo = widget.frameGeometry()
            if bar_geo.intersects(geo):
                exclude = bar_geo.adjusted(0, -CAPTURE_EXCLUDE_MARGIN, 0, CAPTURE_EXCLUDE_MARGIN)
                above_h = max(0, exclude.top() - geo.top())
                below_h = max(0, geo.bottom() - exclude.bottom())
                if below_h >= above_h and below_h > 0:
                    adj = QRect(geo.left(), exclude.bottom() + 1, geo.width(), below_h)
                elif above_h > 0:
                    adj = QRect(geo.left(), geo.top(), geo.width(), above_h)
                else:
                    adj = geo
                return to_physical(adj, scale)
        except Exception:
            pass
    return to_physical(geo, scale)


def _blackout_bar_in_pil(pil: Image.Image, capture_bbox: dict, widget: QtWidgets.QWidget) -> Image.Image:
    """Black out the area where the top bar overlaps the capture bbox, to avoid self-OCR."""
    if not CAPTURE_EXCLUDE_BAR:
        return pil
    try:
        scr = screen_for_widget(widget)
        scale = dpi_scale_for_screen(scr)
        bar_geo = widget.frameGeometry()
        # Physical coords of bar and capture bbox
        bar_left = int(bar_geo.left() * scale)
        bar_top = int(bar_geo.top() * scale)
        bar_right = int(bar_geo.right() * scale)
        bar_bottom = int(bar_geo.bottom() * scale)

        cap_left = int(capture_bbox["left"])
        cap_top = int(capture_bbox["top"])
        cap_right = cap_left + int(capture_bbox["width"]) - 1
        cap_bottom = cap_top + int(capture_bbox["height"]) - 1

        # Intersection
        ix0 = max(cap_left, bar_left)
        iy0 = max(cap_top, bar_top - CAPTURE_EXCLUDE_MARGIN)
        ix1 = min(cap_right, bar_right)
        iy1 = min(cap_bottom, bar_bottom + CAPTURE_EXCLUDE_MARGIN)

        if ix0 <= ix1 and iy0 <= iy1:
            # Map to image coords
            rx0 = ix0 - cap_left
            ry0 = iy0 - cap_top
            rx1 = ix1 - cap_left
            ry1 = iy1 - cap_top
            draw = ImageDraw.Draw(pil)
            draw.rectangle([rx0, ry0, rx1, ry1], fill=(0, 0, 0))
    except Exception:
        pass
    return pil


def pil_from_mss_shot(shot) -> Image.Image:
    return Image.frombytes("RGB", (shot.width, shot.height), shot.rgb)


def dhash(img: Image.Image, size: int = DHASH_SIZE) -> int:
    # Faster implementation with NumPy instead of getpixel
    g = ImageOps.grayscale(img).resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.uint8)
    diff = a[:, 1:] > a[:, :-1]
    val = 0
    for b in diff.flatten():
        val = (val << 1) | int(b)
    return val


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ---------- Embeddings + RAG ----------
@dataclass
class Doc:
    text: str
    ts: float
    h: str


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


class EmbeddingIndex:
    """FAISS + OpenAI embeddings (no Torch)."""

    def __init__(self, model: str = OPENAI_MODEL_EMB, dim: int = 1536):
        self.model = model
        self.dim = dim
        self.index = faiss.IndexFlatIP(self.dim) if faiss is not None else None
        self.vectors = np.zeros((0, self.dim), dtype="float32")
        self.docs: List[Doc] = []
        self._pending: List[Doc] = []  # batch embedding buffer

    def _embed(self, texts: List[str]) -> np.ndarray:
        # Robust embedding: OpenAI if available, else deterministic local
        if not OPENAI_API_KEY or OpenAI is None:
            return _sg_local_embed(texts, self.dim)
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            res = client.embeddings.create(model=self.model, input=texts)
            vecs = np.array([d.embedding for d in res.data], dtype="float32")
            return _normalize(vecs)
        except Exception:
            return _sg_local_embed(texts, self.dim)

    def _add_vectors(self, emb: np.ndarray, docs: List[Doc]):
        if emb.size == 0:
            return
        if self.vectors.shape[0] == 0:
            self.vectors = emb
        else:
            self.vectors = np.vstack([self.vectors, emb])
        if self.index is not None:
            self.index.add(emb)
        self.docs.extend(docs)

    def flush_pending(self):
        if not self._pending:
            return
        texts = [d.text for d in self._pending]
        emb = self._embed(texts)
        self._add_vectors(emb, self._pending)
        self._pending = []

    def add(self, doc: Doc):
        self._pending.append(doc)
        if len(self._pending) >= 4:
            self.flush_pending()

    def prune(self, n: int):
        if len(self.docs) <= n:
            return
        self.flush_pending()
        keep = self.docs[-n:]
        self.index = faiss.IndexFlatIP(self.dim) if faiss is not None else None
        self.vectors = np.zeros((0, self.dim), dtype="float32")
        self.docs = []
        for d in keep:
            self.add(d)
        self.flush_pending()

    def search(self, query: str, top_k: int):
        if len(self.docs) == 0:
            return np.zeros((0, self.dim), dtype="float32"), []
        self.flush_pending()  # ensure the newest docs are in the index
        qv = self._embed([query])
        k = min(top_k, len(self.docs))
        if self.index is not None:
            D, I = self.index.search(qv, k)
            idxs = [i for i in I[0].tolist() if i >= 0]
            return qv, idxs
        # NumPy fallback: cosine similarity over normalized vectors
        if self.vectors.shape[0] == 0:
            return qv, []
        sims = (self.vectors @ qv.T).ravel()
        order = np.argsort(-sims)[:k]
        return qv, [int(i) for i in order]


def mmr_select(qv: np.ndarray, docv: np.ndarray, k: int, lam: float = 0.7) -> List[int]:
    if docv.shape[0] == 0:
        return []
    sim = (docv @ qv.T).ravel()
    selected, cand = [], list(range(docv.shape[0]))
    while len(selected) < min(k, len(cand)):
        if not selected:
            i = int(np.argmax(sim[cand])); selected.append(cand.pop(i)); continue
        red = docv[cand] @ docv[selected].T
        mmr = lam*sim[cand] - (1-lam)*red.max(axis=1)
        i = int(np.argmax(mmr)); selected.append(cand.pop(i))
    return selected


class ContextStore:
    def __init__(self, k: int = CONTEXT_MAX_DOCS):
        self.k = k
        self.docs: List[Doc] = []
        self.emb = EmbeddingIndex()

    def reset(self):
        self.docs = []
        self.emb = EmbeddingIndex()

    def add(self, text: str):
        t = text.strip()
        if not t:
            return
        h = hashlib.md5(t.encode("utf-8")).hexdigest()
        if self.docs and self.docs[-1].h == h:
            return
        d = Doc(t, time.time(), h)
        self.docs.append(d); self.emb.add(d); self.emb.prune(self.k)

    def retrieve(self, query: str, top_n: int = CONTEXT_TOP_K) -> List[Doc]:
        # Always bias toward the most recent context
        recent: List[Doc] = self.docs[-1:] if self.docs else []
        qv, idxs = self.emb.search(query, top_k=max(top_n*2, top_n))
        if not idxs:
            return (self.docs[-(top_n-1):] + recent) if top_n > 1 else (recent or [])
        cand = self.emb.vectors[idxs]
        order = mmr_select(qv, cand, k=max(1, top_n-1), lam=0.7)
        sel = [idxs[i] for i in order]
        picked = [self.emb.docs[i] for i in sel]
        # Merge with recent, de-duplicated, recent first
        out: List[Doc] = []
        seen = set()
        for d in (recent + picked):
            if d.h in seen:
                continue
            out.append(d)
            seen.add(d.h)
            if len(out) >= top_n:
                break
        return out


CTX = ContextStore()


def _sg_local_embed(texts: List[str], dim: int) -> np.ndarray:
    out: List[np.ndarray] = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        v = rng.normal(0, 1, dim).astype("float32")
        n = float(np.linalg.norm(v) + 1e-9)
        out.append(v / n)
    return np.vstack(out) if out else np.zeros((0, dim), dtype="float32")



# Strengthened prompt to avoid disclaimers about not seeing the screen

def sanitize_llm_answer(ans: str) -> str:
    try:
        lines = ans.splitlines()
        out = []
        skip = [
            "i can't see your screen",
            "i cannot see your screen",
            "i don't have access to your screen",
            "i dont have access to your screen",
            "i don't have the ability to view your screen",
            "i cant see your screen",
            "as an ai",
            "i can't view images",
            "i cannot view images",
            "i can't view your screen",
            "i cannot view your screen",
        ]
        for line in lines:
            lo = line.strip().lower()
            if any(ph in lo for ph in skip):
                continue
            out.append(line)
        cleaned = "\n".join(out).strip()
        for ph in skip:
            cleaned = cleaned.replace(ph, "based on the OCR context")
        return cleaned or ans
    except Exception:
        return ans
def call_llm(question: str, context_blob: str) -> str:
    """Chat call with graceful fallback if key/model unavailable."""
    if not OPENAI_API_KEY or OpenAI is None:
        # Local fallback that still echoes context to be useful offline
        ctx_lines = [f"- {l}" for l in context_blob.splitlines() if l.strip()]
        return ("(offline mode) " + question + "\n" + "\n".join(ctx_lines))[:1200]
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        sys_prompt = (
            "You are a helpful assistant. You are given OCR text captured from the user's screen as 'Context (OCR)'. "
            "Treat it as if you can see their screen. Do NOT say 'I cannot see your screen'. "
            "Use only the provided context to reason and answer in English. "
            "If the context is insufficient, ask a clarifying question rather than claiming you cannot see the screen."
        )
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": f"Context (OCR):\n{context_blob}\n\nQuestion: {question}"},
        ]
        r = client.chat.completions.create(model=OPENAI_MODEL_CHAT, messages=msgs, temperature=0.2)
        content = (r.choices[0].message.content or "").strip()
        return sanitize_llm_answer(content)
    except Exception as e:
        return f"LLM call error: {e}"
class AnswerBubble(QtWidgets.QWidget):
    """Dark, draggable answer bubble below the top bar. Stays until closed."""

    def __init__(self, parent_bar: QtWidgets.QWidget):
        super().__init__(parent=parent_bar)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.box = QtWidgets.QFrame(self)
        self.box.setStyleSheet(
            "QFrame { background: rgba(22,22,26,220); border-radius: 12px; } "
            "QTextEdit { background: transparent; color: #f2f2f2; border: none; padding: 10px; }"
        )
        v = QtWidgets.QVBoxLayout(self.box)
        v.setContentsMargins(8, 8, 8, 8)
        self.text = QtWidgets.QTextEdit(readOnly=True)
        self.text.setMinimumHeight(80)
        self.text.setMaximumHeight(260)
        v.addWidget(self.text)
        # Close button in top-right of bubble
        self.closeBtn = QtWidgets.QPushButton("X", self)
        self.closeBtn.setToolTip("Close answer")
        self.closeBtn.setStyleSheet("QPushButton { color:white; background:transparent; border:none; font-weight:bold; }")
        self.closeBtn.clicked.connect(self.fade_out_and_hide)


        self.resize(640, 160)
        self.box.setGeometry(0, 0, self.width(), self.height())
        self._dragging = False
        self._drag_offset = QPoint()

        self._eff = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._eff)
        self._anim = QtCore.QPropertyAnimation(self._eff, b"opacity", self)
        self._anim.setDuration(180)

        self.hide()


    def show_answer(self, s: str, duration_ms: int = 0):
        # Ensure no stale auto-hide connection remains and show instantly (no auto-hide)
        try:
            self._anim.stop()
        except Exception:
            pass
        try:
            self._anim.finished.disconnect()
        except Exception:
            pass
        if not self.isVisible():
            parent = self.parentWidget()
            if parent:
                x = parent.width()//2 - self.width()//2
                y = parent.height() + 8
                self.move(max(8, x), y)
        self.text.setHtml(s.replace("\n", "<br>"))
        try:
            self._eff.setOpacity(1.0)
        except Exception:
            pass
        self.show()
        self.raise_()

    def _place_close_btn(self):
        try:
            self.closeBtn.move(self.width() - 28, 6)
        except Exception:
            pass

    def fade_out_and_hide(self):
        self._anim.stop()
        self._anim.setStartValue(self._eff.opacity())
        self._anim.setEndValue(0.0)
        try:
            self._anim.finished.disconnect()
        except Exception:
            pass
        self._anim.finished.connect(self.hide)
        self._anim.start()

    # draggable bubble
    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_offset = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            e.accept()
        else:
            e.ignore()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self._dragging:
            self.move(e.globalPosition().toPoint() - self._drag_offset)
            self._place_close_btn()
            e.accept()
        else:
            e.ignore()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if self._dragging and e.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            e.accept()
        else:
            e.ignore()

    def resizeEvent(self, _e):
        self.box.setGeometry(0, 0, self.width(), self.height())
        self._place_close_btn()


class MessageBubble(QtWidgets.QFrame):
    def __init__(self, role: str, html: str, timestamp: str, parent=None):
        super().__init__(parent)
        self.setObjectName('MsgBubble')
        # Base style; role-specific backgrounds applied below
        self.setStyleSheet(
            "QFrame#MsgBubble { border-radius: 18px; }\n"
            "QTextBrowser { border:none; background:transparent; margin:0px; padding:0px;" \
            " font-family:'Segoe UI', Inter, Roboto, Arial, 'Helvetica Neue', sans-serif; font-size:13px; line-height:1.35; color:#FFFFFF; }\n"
            "QTextBrowser:hover { background:transparent; }"
        )
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 8)
        lay.setSpacing(6)

        # Subtle drop shadow for depth
        _shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        _shadow.setBlurRadius(18)
        _shadow.setColor(QtGui.QColor(0, 0, 0, 76))
        _shadow.setOffset(0, 2)
        self.setGraphicsEffect(_shadow)

        self.browser = QtWidgets.QTextBrowser(self)
        self.browser.setOpenExternalLinks(True)
        self.browser.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.browser.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.browser.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.browser.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)
        self.browser.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.browser.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse |
            QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self.browser.setHtml(html)
        try:
            _doc = self.browser.document()
            _doc.setTextWidth(max(100.0, float(self.maximumWidth() - 32)))
            _h = int(_doc.size().height()) + 4
            self.browser.setMinimumHeight(_h)
            self.browser.setMaximumHeight(_h)
        except Exception:
            pass
        lay.addWidget(self.browser)

        self.ts = QtWidgets.QLabel(timestamp, self)
        f = self.ts.font(); f.setPointSize(10); self.ts.setFont(f)
        self.ts.setStyleSheet("color:#CCCCCC; margin-top:3px;")
        lay.addWidget(self.ts, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        # Role-specific background colors
        if role == 'user':
            self.setStyleSheet(self.styleSheet() + "QFrame#MsgBubble{ background:#1877F2; }")
        else:
            self.setStyleSheet(self.styleSheet() + "QFrame#MsgBubble{ background:#2B2B2B; }")

    def set_max_width(self, w: int):
        """Set maximum width and auto-fit the inner browser height to its content.
        Keeps bubbles compact with no internal scrollbars.
        """
        try:
            self.setMaximumWidth(w)
            doc = self.browser.document()
            doc.setTextWidth(max(50.0, float(w - 32)))
            h = int(doc.size().height()) + 4
            self.browser.setMinimumHeight(h)
            self.browser.setMaximumHeight(h)
        except Exception:
            pass


class HistoryWindow(QtWidgets.QWidget):
    """Modern chat-like history with aligned message bubbles and manual save only."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Answer History")
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.resize(860, 640)
        self._plain_log: List[str] = []
        self._bubbles: list[MessageBubble] = []

        # Overall dark theme background
        self.setStyleSheet(
            "QWidget { background:#1E1E1E; color:#EAEAEA; }\n"
            "QScrollArea { border:none; background:#1E1E1E; }\n"
        )
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 16)
        outer.setSpacing(10)

        head = QtWidgets.QHBoxLayout(); outer.addLayout(head)
        title = QtWidgets.QLabel("History")
        tf = title.font(); tf.setPointSize(11); tf.setBold(True); title.setFont(tf)
        head.addWidget(title); head.addStretch(1)
        self.btnSave = QtWidgets.QPushButton("Save...")
        self.btnSave.setStyleSheet("QPushButton{background:#1f2430; color:#EAEAEA; border:1px solid #2a2f3a; border-radius:8px; padding:6px 12px;} QPushButton:hover{background:#293142;}")
        self.btnSave.clicked.connect(self.save_dialog)
        head.addWidget(self.btnSave)

        self.scroll = QtWidgets.QScrollArea(self); self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll, 1)
        self._host = QtWidgets.QWidget()
        self._list = QtWidgets.QVBoxLayout(self._host)
        self._list.setContentsMargins(12, 12, 12, 12)
        self._list.setSpacing(10)
        self._list.addStretch(1)
        self.scroll.setWidget(self._host)
        # small spacing under header
        outer.addSpacing(8)

    def _format_html(self, text: str) -> str:
        # Escape text and render simple markdown-like parts and ``` code blocks
        def esc(s: str) -> str:
            return (s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'))
        t = text.replace('\r','')
        blocks = t.split('```')
        out: list[str] = []
        for i, b in enumerate(blocks):
            if i % 2 == 1:
                out.append(f"<pre style=\"background:#0F0F12;border:1px solid #202026;border-radius:12px;padding:8px;\"><code style=\"font-family:Consolas,'Courier New',monospace;font-size:12px;\">{esc(b)}</code></pre>")
            else:
                lines = b.split('\n')
                ul = False
                for line in lines:
                    ls = line.strip()
                    if ls.startswith(('#','##','###')):
                        lvl = len(ls) - len(ls.lstrip('#'))
                        txt = esc(ls.lstrip('#').strip())
                        size = 16 if lvl==1 else 14 if lvl==2 else 13
                        out.append(f"<div style=\"font-weight:600;font-size:{size}px;margin:6px 0;\">{txt}</div>")
                    elif ls.startswith(('- ','* ')):
                        if not ul:
                            out.append('<ul style="margin:4px 16px;">'); ul = True
                        out.append(f"<li style=\"margin:2px 0;\">{esc(ls[2:])}</li>")
                    else:
                        if ul:
                            out.append('</ul>'); ul=False
                        if ls:
                            out.append(f"<div style=\"line-height:1.35;\">{esc(line)}</div>")
                        else:
                            out.append('<div style="height:6px;"></div>')
                if ul:
                    out.append('</ul>')
        return "\n".join(out)

    def _max_bubble_width(self) -> int:
        vw = self.scroll.viewport().width() if self.scroll and self.scroll.viewport() else self.width()
        return int(max(320, vw * 0.55))
    def resizeEvent(self, e: QtGui.QResizeEvent):
        super().resizeEvent(e)
        mw = self._max_bubble_width()
        for b in self._bubbles:
            try:
                b.set_max_width(mw)
            except Exception:
                b.setMaximumWidth(mw)

    def _add_bubble(self, role: str, text: str):
        wrap = QtWidgets.QWidget(self._host)
        row = QtWidgets.QHBoxLayout(wrap); row.setContentsMargins(6,6,6,6); row.setSpacing(6)
        ts = datetime.datetime.now().strftime('%H:%M')
        html = self._format_html(text)
        bubble = MessageBubble(role, html, ts, parent=wrap)
        bubble.set_max_width(self._max_bubble_width())
        if role == 'user':
            row.addStretch(1); row.addWidget(bubble)
        else:
            row.addWidget(bubble); row.addStretch(1)
        idx = self._list.count()-1
        self._list.insertWidget(idx, wrap)
        self._bubbles.append(bubble)
        # Autoscroll to bottom
        QtCore.QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()))
        # Plain text log
        self._plain_log.append(text)

    @QtCore.pyqtSlot(str)
    def append_user(self, text: str):
        self._add_bubble('user', text)

    @QtCore.pyqtSlot(str)
    def append_ai(self, text: str):
        self._add_bubble('ai', text)

    def append_html(self, html: str):
        # Compatibility: coerce incoming HTML to readable text then render as AI bubble
        try:
            txt = QtGui.QTextDocumentFragment.fromHtml(html).toPlainText()
        except Exception:
            txt = html
        self._add_bubble('ai', txt)

    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self.show(); self.raise_(); self.activateWindow()

    def save_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save history", os.path.join("C:\\ai", "history.txt"), "Text Files (*.txt);;All Files (*)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(self._plain_log))
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Save error", str(e))

    def auto_export_on_quit(self):
        # Disabled: manual save only
        return

    def clear_all(self):
        try:
            while self._list.count() > 1:
                itm = self._list.takeAt(0)
                w = itm.widget()
                if w is not None:
                    w.setParent(None)
            self._plain_log = []
            self._bubbles = []
        except Exception:
            pass


class SettingsDialog(QtWidgets.QDialog):
    """Small settings dialog for performance - keeps the top bar design unchanged."""
    def __init__(self, parent=None, apply_cb=None):
        super().__init__(parent)
        self.setWindowTitle("Settings - Performance")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.apply_cb = apply_cb
        form = QtWidgets.QFormLayout(self)

        self.inp_langs = QtWidgets.QLineEdit()
        self.inp_whitelist = QtWidgets.QLineEdit()
        self.chk_psm_auto = QtWidgets.QCheckBox("Automatic PSM (faster for narrow fragments)")

        self.spin_base = QtWidgets.QDoubleSpinBox(); self.spin_base.setRange(0.2, 20.0); self.spin_base.setSingleStep(0.2)
        self.spin_eco_max = QtWidgets.QDoubleSpinBox(); self.spin_eco_max.setRange(0.5, 60.0); self.spin_eco_max.setSingleStep(0.5)
        self.spin_eco_step = QtWidgets.QDoubleSpinBox(); self.spin_eco_step.setRange(0.1, 10.0); self.spin_eco_step.setSingleStep(0.1)
        self.spin_eco_ticks = QtWidgets.QSpinBox(); self.spin_eco_ticks.setRange(1, 20)

        self.spin_min_pixels = QtWidgets.QSpinBox(); self.spin_min_pixels.setRange(10_000, 5_000_000); self.spin_min_pixels.setSingleStep(10_000)
        self.spin_dhash_thr = QtWidgets.QSpinBox(); self.spin_dhash_thr.setRange(1, 32)

        self.cmb_capture = QtWidgets.QComboBox(); self.cmb_capture.addItems(["full","center","custom"])
        self.inp_bbox = QtWidgets.QLineEdit(); self.inp_bbox.setPlaceholderText("left,top,width,height")

        form.addRow("OCR languages (e.g., pol+eng)", self.inp_langs)
        form.addRow("Whitelist (optional)", self.inp_whitelist)
        form.addRow(self.chk_psm_auto)
        form.addRow("Base interval (s)", self.spin_base)
        form.addRow("Eco max (s)", self.spin_eco_max)
        form.addRow("Eco step (s)", self.spin_eco_step)
        form.addRow("Eco stable ticks", self.spin_eco_ticks)
        form.addRow("Minimum area (px)", self.spin_min_pixels)
        form.addRow("DHash threshold", self.spin_dhash_thr)
        form.addRow("Capture area", self.cmb_capture)
        form.addRow("Custom bbox", self.inp_bbox)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Apply)
        form.addRow(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(self.on_apply)

        self.resize(460, 0)

    def load_from_globals(self):
        self.inp_langs.setText(OCR_LANGS)
        self.inp_whitelist.setText(OCR_WHITELIST)
        self.chk_psm_auto.setChecked(PSM_AUTOTUNE)
        self.spin_base.setValue(float(LIVE_OCR_INTERVAL_SEC))
        self.spin_eco_max.setValue(float(ECO_MAX_INTERVAL_SEC))
        self.spin_eco_step.setValue(float(ECO_STEP_SEC))
        self.spin_eco_ticks.setValue(int(ECO_STABLE_TICKS_TO_STEP))
        self.spin_min_pixels.setValue(int(AUTO_OCR_MIN_PIXELS))
        self.spin_dhash_thr.setValue(int(DHASH_HAMMING_THRESHOLD))
        self.cmb_capture.setCurrentText(CAPTURE_MODE)
        if CAPTURE_CUSTOM_BBOX and CAPTURE_CUSTOM_BBOX.get("width",0)>0:
            self.inp_bbox.setText(
                f"{CAPTURE_CUSTOM_BBOX['left']},{CAPTURE_CUSTOM_BBOX['top']},{CAPTURE_CUSTOM_BBOX['width']},{CAPTURE_CUSTOM_BBOX['height']}")

    def _collect(self) -> dict:
        bbox = {"left":0,"top":0,"width":0,"height":0}
        if self.cmb_capture.currentText()=="custom" and self.inp_bbox.text().strip():
            try:
                parts = [int(x.strip()) for x in self.inp_bbox.text().split(",")]
                if len(parts)==4:
                    bbox = {"left":parts[0],"top":parts[1],"width":parts[2],"height":parts[3]}
            except Exception:
                pass
        return {
            "OCR_LANGS": self.inp_langs.text(),
            "OCR_WHITELIST": self.inp_whitelist.text(),
            "PSM_AUTOTUNE": self.chk_psm_auto.isChecked(),
            "LIVE_OCR_INTERVAL_SEC": self.spin_base.value(),
            "ECO_MAX_INTERVAL_SEC": self.spin_eco_max.value(),
            "ECO_STEP_SEC": self.spin_eco_step.value(),
            "ECO_STABLE_TICKS_TO_STEP": self.spin_eco_ticks.value(),
            "AUTO_OCR_MIN_PIXELS": self.spin_min_pixels.value(),
            "DHASH_HAMMING_THRESHOLD": self.spin_dhash_thr.value(),
            "CAPTURE_MODE": self.cmb_capture.currentText(),
            "CAPTURE_CUSTOM_BBOX": bbox,
        }

    def on_apply(self):
        if self.apply_cb:
            self.apply_cb(self._collect())

    def accept(self):
        self.on_apply()
        super().accept()


class TopBar(QtWidgets.QWidget):
    askSignal = QtCore.pyqtSignal(str)
    liveToggleSignal = QtCore.pyqtSignal(bool)
    ecoToggleSignal  = QtCore.pyqtSignal(bool)
    historyToggleSignal = QtCore.pyqtSignal()
    positionChanged = QtCore.pyqtSignal()
    settingsRequested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Overlay Q&A - TopBar")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedHeight(TOPBAR_HEIGHT)

        box = QtWidgets.QFrame(self)
        box.setStyleSheet(
            "QFrame { background: rgba(22,22,26,190); border-radius: 12px; } "
            "QLineEdit { background: rgba(255,255,255,28); border: 1px solid rgba(255,255,255,50);"
            " border-radius: 8px; padding: 6px 10px; color: white;} "
            "QPushButton { background: rgba(255,255,255,40); border: none; border-radius: 8px; padding: 6px 10px; color: white;} "
            "QPushButton:hover { background: rgba(255,255,255,60);} "
            "QLabel { color: #ddd; }"
        )
        h = QtWidgets.QHBoxLayout(box)
        h.setContentsMargins(12,8,12,8)
        h.setSpacing(8)

        self.handle = QtWidgets.QLabel(":::")
        self.inp = QtWidgets.QLineEdit(); self.inp.setPlaceholderText("Ask a question...")
        self.btnLive = QtWidgets.QPushButton("Live OCR"); self.btnLive.setCheckable(True)
        self.btnEco  = QtWidgets.QPushButton("Eco"); self.btnEco.setCheckable(True)
        self.btnHist = QtWidgets.QPushButton("History")
        self.btnSettings = QtWidgets.QPushButton("Settings")
        self.btnAsk  = QtWidgets.QPushButton("Send")
        self.btnClose= QtWidgets.QPushButton("X")
        self.btnClose.setToolTip("Close")
        self.btnClose.setFixedWidth(30)
        self.btnClose.setStyleSheet(
            "QPushButton { background: rgba(255,255,255,40); border-radius: 10px; color: white; font-weight: bold; } "
            "QPushButton:hover { background: rgba(255,80,80,80); }"
        )
        try: self.btnClose.setAutoDefault(False)
        except Exception: pass
        try: self.btnClose.setDefault(False)
        except Exception: pass
        try:
            self.btnAsk.setAutoDefault(True)
            self.btnAsk.setDefault(True)
        except Exception:
            pass

        h.addWidget(self.handle)
        h.addWidget(self.inp, 1)
        h.addWidget(self.btnLive)
        h.addWidget(self.btnEco)
        h.addWidget(self.btnHist)
        h.addWidget(self.btnSettings)
        h.addWidget(self.btnAsk)
        h.addWidget(self.btnClose)

        box.setGeometry(0,0,TOPBAR_WIDTH,TOPBAR_HEIGHT)
        self.setFixedWidth(TOPBAR_WIDTH)

        scr = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.move(scr.center().x() - self.width()//2, scr.top() + 6)

        self.btnAsk.clicked.connect(self._emit_ask)
        self.inp.returnPressed.connect(self._emit_ask)
        self.btnLive.toggled.connect(lambda s: self.liveToggleSignal.emit(s))
        self.btnEco.toggled.connect(lambda s: self.ecoToggleSignal.emit(s))
        self.btnHist.clicked.connect(lambda: self.historyToggleSignal.emit())
        self.btnSettings.clicked.connect(lambda: self.settingsRequested.emit())
        self.btnClose.clicked.connect(QtWidgets.QApplication.quit)

        self._drag=False; self._drag_pos=QPoint()

        self.scLive = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+L"), self)
        try: self.scLive.setContext(Qt.ShortcutContext.ApplicationShortcut)
        except Exception: pass
        self.scLive.activated.connect(self._on_live_shortcut)

        self.scHist = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+H"), self)
        try: self.scHist.setContext(Qt.ShortcutContext.ApplicationShortcut)
        except Exception: pass
        self.scHist.activated.connect(lambda: self.historyToggleSignal.emit())

        # Settings shortcuts: Ctrl+Comma, Std Preferences, and Ctrl+Alt+S
        self.scSettings1 = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+,"), self)
        self.scSettings2 = QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Preferences), self)
        self.scSettings3 = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Alt+S"), self)
        for sc in (self.scSettings1, self.scSettings2, self.scSettings3):
            try: sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            except Exception: pass
            sc.activated.connect(lambda: self.settingsRequested.emit())

    def _on_live_shortcut(self):
        self.btnLive.setChecked(not self.btnLive.isChecked())
        self.liveToggleSignal.emit(self.btnLive.isChecked())

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = True
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self._drag:
            self.move(e.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, _e: QtGui.QMouseEvent):
        self._drag = False
        try:
            self.positionChanged.emit()
        except Exception:
            pass

    def _emit_ask(self):
        q = self.inp.text().strip()
        if q:
            self.askSignal.emit(q)
            self.inp.clear()


class ForegroundWatcher(QtCore.QObject):
    changed = QtCore.pyqtSignal(tuple)

    def __init__(self, interval_ms: int = 300, parent=None):
        super().__init__(parent)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(max(100, int(interval_ms)))
        self._timer.timeout.connect(self._poll)
        self._last: tuple = (0, "", 0)

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def _active_window_sig(self) -> tuple:
        try:
            user32 = ctypes.windll.user32
            hwnd = int(user32.GetForegroundWindow())
            if hwnd == 0:
                return (0, "", 0)
            length = int(user32.GetWindowTextLengthW(hwnd))
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value
            pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            return (hwnd, title, int(pid.value))
        except Exception:
            return self._last

    def _poll(self):
        sig = self._active_window_sig()
        if sig != self._last and sig[0] != 0:
            self._last = sig
            try:
                self.changed.emit(sig)
            except Exception:
                pass


class Toast(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)
        # Top-most tool window that does not steal focus
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowStaysOnTopHint
        # Use window opacity for translucency (more reliable in packaged EXE)
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setWindowOpacity(0.92)
        self.lab = QtWidgets.QLabel("", self)
        self.lab.setStyleSheet("QLabel{background:transparent; color:white;}")
        self._pad_h = 14
        self._pad_v = 10

    def paintEvent(self, _e: QtGui.QPaintEvent):
        # Draw dark rounded background behind the label
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setBrush(QtGui.QColor(0, 0, 0, 255))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(self.rect(), 12, 12)

    def show_text(self, text: str, ms: int = 1400):
        self.lab.setText(text)
        self.lab.adjustSize()
        # Pick screen under the bar or cursor for correct multi-monitor placement
        scr = None
        try:
            p = self.parentWidget()
            if p is not None:
                scr = p.screen()
        except Exception:
            scr = None
        if scr is None:
            scr = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos()) or QtGui.QGuiApplication.primaryScreen()
        geo = scr.availableGeometry()
        w = self.lab.width() + self._pad_h * 2
        h = self.lab.height() + self._pad_v * 2
        self.resize(w, h)
        self.lab.move(self._pad_h, self._pad_v)
        x = geo.right() - w - 24
        y = geo.bottom() - h - 24
        self.move(max(geo.left()+8, x), max(geo.top()+8, y))
        self.setWindowOpacity(0.92)
        self.show()
        self.raise_()
        QTimer.singleShot(ms, self.hide)


# ---------- Worker OCR in a background thread
class OcrWorker(QtCore.QObject):
    ocrText   = QtCore.pyqtSignal(str)   # OCR text ready
    toast     = QtCore.pyqtSignal(str)   # light notifications
    busy      = QtCore.pyqtSignal(bool)  # to drive a spinner if needed

    def __init__(self, bbox: dict, parent=None):
        super().__init__(parent)
        self._bbox = dict(bbox)
        self._stop = False
        self._live = False
        self._eco_enabled = False
        self._base_interval = LIVE_OCR_INTERVAL_SEC
        self._current_interval = self._base_interval
        self._stable_ticks = 0
        self._sct = None  # created at run()
        self._bar_widget: Optional[QtWidgets.QWidget] = None

    @QtCore.pyqtSlot(dict)
    def set_bbox(self, bbox: dict):
        self._bbox = dict(bbox)

    @QtCore.pyqtSlot(float, float, int)
    def set_eco_params(self, base_interval: float, eco_max: float, stable_ticks: int):
        self._base_interval = max(0.2, float(base_interval))
        self._current_interval = self._base_interval
        global ECO_MAX_INTERVAL_SEC, ECO_STABLE_TICKS_TO_STEP
        ECO_MAX_INTERVAL_SEC = float(eco_max)
        ECO_STABLE_TICKS_TO_STEP = int(stable_ticks)

    @QtCore.pyqtSlot(QtWidgets.QWidget)
    def set_bar_widget(self, w: QtWidgets.QWidget):
        self._bar_widget = w

    @QtCore.pyqtSlot()
    def reset_ocr(self):
        self._stable_ticks = 0
        self._current_interval = self._base_interval

    @QtCore.pyqtSlot()
    def run(self):
        # Optional perf CSV header
        log_path = None
        if PERF_LOG_CSV:
            home = os.path.expanduser("~")
            d = os.path.join(home, ".studyglance")
            os.makedirs(d, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d")
            log_path = os.path.join(d, f"perf-overlay-{ts}.csv")
            if not os.path.exists(log_path):
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("ts_ms,capture_ms,ocr_ms,total_ms\n")

        def perf(ts_ms, cap, ocr, tot):
            if not log_path:
                return
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{ts_ms},{cap:.2f},{ocr:.2f},{tot:.2f}\n")
            except Exception:
                pass

        self._sct = mss()
        self.toast.emit(f"Worker started (interval {self._current_interval:.1f}s)")
        try:
            while not self._stop:
                if not self._live:
                    time.sleep(0.05)
                    continue

                t0 = time.perf_counter_ns()
                self.busy.emit(True)

                bbox = self._bbox
                t_cap0 = time.perf_counter_ns()
                shot = self._sct.grab(bbox)
                pil = pil_from_mss_shot(shot)
                if self._bar_widget is not None:
                    pil = _blackout_bar_in_pil(pil, bbox, self._bar_widget)
                cap_ms = (time.perf_counter_ns() - t_cap0) / 1e6

                if pil.width * pil.height < AUTO_OCR_MIN_PIXELS:
                    self.busy.emit(False)
                    time.sleep(self._current_interval)
                    continue

                # Always OCR each tick (simple and reliable)
                t_ocr0 = time.perf_counter_ns()
                txt = ocr_image(pil)
                ocr_ms = (time.perf_counter_ns() - t_ocr0) / 1e6
                if txt:
                    self.ocrText.emit(txt)
                    self.toast.emit("OCR updated")
                if self._eco_enabled and self._current_interval > LIVE_OCR_INTERVAL_SEC:
                    self._current_interval = LIVE_OCR_INTERVAL_SEC
                    self._stable_ticks = 0

                self.busy.emit(False)
                total_ms = (time.perf_counter_ns() - t0) / 1e6
                perf(int(time.time() * 1000), cap_ms, ocr_ms, total_ms)
                time.sleep(self._current_interval)
        except Exception as e:
            print(f"[OcrWorker] error: {e}", file=sys.stderr)
        finally:
            try:
                if self._sct:
                    self._sct.close()
            except Exception:
                pass
            self.toast.emit("Worker stopped")

    @QtCore.pyqtSlot(bool)
    def set_live(self, on: bool):
        self._live = on
        if on:
            self._stable_ticks = 0
            self._current_interval = self._base_interval
            try:
                b = self._bbox
                self.toast.emit(f"Live OCR: ON ({self._current_interval:.1f}s) bbox={b['left']},{b['top']} {b['width']}x{b['height']}")
            except Exception:
                self.toast.emit(f"Live OCR: ON ({self._current_interval:.1f}s)")
            # Seed context immediately with a one-shot OCR
            try:
                if self._sct is None:
                    self._sct = mss()
                shot = self._sct.grab(self._bbox)
                pil = pil_from_mss_shot(shot)
                if self._bar_widget is not None:
                    pil = _blackout_bar_in_pil(pil, self._bbox, self._bar_widget)
                txt = ocr_image(pil)
                if txt:
                    self.ocrText.emit(txt)
                    self.toast.emit("OCR updated")
            except Exception:
                pass
        else:
            self.toast.emit("Live OCR: OFF")

    @QtCore.pyqtSlot(bool)
    def set_eco(self, on: bool):
        self._eco_enabled = on
        if on:
            self.toast.emit("Eco: ON (dynamic interval)")
        else:
            self._current_interval = self._base_interval
            self._stable_ticks = 0
            self.toast.emit("Eco: OFF")

    @QtCore.pyqtSlot()
    def stop(self):
        self._stop = True

class Core(QtCore.QObject):
    toastSignal = QtCore.pyqtSignal(str)
    showAnswerSignal = QtCore.pyqtSignal(str)
    historySignal = QtCore.pyqtSignal(str)

    def __init__(self, bar: "TopBar", bubble: "AnswerBubble", hist: "HistoryWindow"):
        super().__init__()
        self.bar = bar
        self.bubble = bubble
        self.hist = hist
        self._worker: Optional[OcrWorker] = None
        self._capture_allowed = False
        self._silence_refresh_once = False

        # Global keyboard shortcut for Settings (Ctrl+,)
        self._settings = None
        self._scSettings = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+,"), self.bar)
        try:
            self._scSettings.setContext(Qt.ShortcutContext.ApplicationShortcut)
        except Exception:
            pass
        self._scSettings.activated.connect(self.show_settings)

        # Manual reset hotkey (Ctrl+R)
        try:
            self._scReset = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self.bar)
            self._scReset.setContext(Qt.ShortcutContext.ApplicationShortcut)
            self._scReset.activated.connect(self.reset_all)
        except Exception:
            pass

        self._reset_timer: Optional[QtCore.QTimer] = None

    @QtCore.pyqtSlot(str)
    def on_ocr_text(self, txt: str):
        CTX.add(txt)

    def ask(self, q: str):
        try:
            if not tesseract_ok():
                try: self.toastSignal.emit("Tesseract not found. Install Tesseract or set TESSERACT_CMD.")
                except Exception: pass
            
            # One-shot seed from screen regardless of live state (diagnostic toasts)
            try:
                bbox = capture_screen_bbox_of_widget(self.bar)
                with mss() as _sct:
                    shot = _sct.grab(bbox)
                pil = pil_from_mss_shot(shot)
                pil = _blackout_bar_in_pil(pil, bbox, self.bar)
                txt = ocr_image(pil)
                if txt:
                    CTX.add(txt)
                    try: self.toastSignal.emit(f"Seeded context ({len(txt)} chars)")
                    except Exception: pass
                else:
                    try: self.toastSignal.emit("Seed capture found no text")
                    except Exception: pass
            except Exception as e:
                try: self.toastSignal.emit(f"Seed OCR error: {e}")
                except Exception: pass
            # Seed capture (debug): save images and OCR without blackout
            try:
                bbox = capture_screen_bbox_of_widget(self.bar)
                with mss() as _sct:
                    shot = _sct.grab(bbox)
                pil = pil_from_mss_shot(shot)
                try:
                    pil.save(r"C:\overlay_ai\debug_seed.png")
                except Exception:
                    pass
                prep = preprocess_for_ocr(pil)
                try:
                    prep.save(r"C:\overlay_ai\debug_seed_pre.png")
                except Exception:
                    pass
                txt_dbg = ocr_image(pil)
                if txt_dbg:
                    CTX.add(txt_dbg)
                    try: self.toastSignal.emit(f"Seeded context ({len(txt_dbg)} chars)")
                    except Exception: pass
                else:
                    try: self.toastSignal.emit("Seed capture found no text")
                    except Exception: pass
            except Exception as e:
                try: self.toastSignal.emit(f"Seed OCR error: {e}")
                except Exception: pass
            docs = CTX.retrieve(q, top_n=CONTEXT_TOP_K)
            if not docs:
                msg = "No context. Turn on Live OCR (Ctrl+L), make sure text is visible on screen, then ask again."
                html = f"<b>Answer:</b><br>{msg}"
                self.showAnswerSignal.emit(html)
                self.toastSignal.emit("No context - turn on Live OCR")
                try:
                    self.hist.append_user(q)
                    self.hist.append_ai(msg)
                except Exception:
                    self.historySignal.emit(f"<b>You:</b> {q}<br>{html}<hr>")
                return
            ctx_blob = "\n\n---\n".join(d.text for d in docs)
            ans = call_llm(q, ctx_blob) or "(no answer)"
            safe = ans.replace("<","&lt;").replace(">","&gt;")
            html = f"<b>Answer:</b><br>{safe}"
            self.showAnswerSignal.emit(html)
            self.toastSignal.emit("Answer ready")
            try:
                self.hist.append_user(q)
                self.hist.append_ai(ans)
            except Exception:
                self.historySignal.emit(f"<b>You:</b> {q}<br>{html}<hr>")
            return
        except Exception as e:
            try:
                self.toastSignal.emit(f"Error: {e}")
            except Exception:
                pass
            try:
                self.hist.append_ai(f"Error: {e}")
            except Exception:
                pass
            return
    @QtCore.pyqtSlot()
    def reset_all(self, silent: bool = False):
        # Soft reset: clear OCR/RAG context, keep history window entries
        if self._worker is not None:
            try:
                self._worker.reset_ocr()
            except Exception:
                pass
        try:
            CTX.reset()
        except Exception:
            pass
        if not silent:
            self.toastSignal.emit("Context reset")
        else:
            self._silence_refresh_once = True
        if self._reset_timer is None:
            self._reset_timer = QtCore.QTimer(self)
            self._reset_timer.setSingleShot(True)
            self._reset_timer.timeout.connect(self._seed_context_from_screen)
        if self._capture_allowed:
            self._reset_timer.start(400)
    @QtCore.pyqtSlot()
    def _seed_context_from_screen(self):
        if not self._capture_allowed:
            return
        try:
            bbox = capture_screen_bbox_of_widget(self.bar)
            with mss() as _sct:
                shot = _sct.grab(bbox)
            pil = pil_from_mss_shot(shot)
            pil = _blackout_bar_in_pil(pil, bbox, self.bar)
            txt = ocr_image(pil)
            if not txt:
                # Fallback: full-screen capture
                try:
                    with mss() as _sct:
                        mon = _sct.monitors[0]
                        shot_fs = _sct.grab(mon)
                    pil_fs = pil_from_mss_shot(shot_fs)
                    txt = ocr_image(pil_fs)
                except Exception:
                    txt = ""
            if txt:
                CTX.add(txt)
                if not self._silence_refresh_once:
                    self.toastSignal.emit("Context refreshed")
                else:
                    self._silence_refresh_once = False
        except Exception:
            pass
    @QtCore.pyqtSlot(tuple)
    def on_window_changed(self, _sig: tuple):
        self.reset_all(True)

    def show_settings(self):
        # Modal dialog without changing top bar design
        if self._settings is None:
            self._settings = SettingsDialog(parent=self.bar, apply_cb=self.apply_settings)
        self._settings.load_from_globals()
        self._settings.show(); self._settings.raise_(); self._settings.activateWindow()

    def apply_settings(self, cfg: dict):
        # Update globals
        global OCR_LANGS, LIVE_OCR_INTERVAL_SEC, ECO_MAX_INTERVAL_SEC, ECO_STEP_SEC, ECO_STABLE_TICKS_TO_STEP
        global AUTO_OCR_MIN_PIXELS, DHASH_HAMMING_THRESHOLD, OCR_WHITELIST, PSM_AUTOTUNE, CAPTURE_MODE, CAPTURE_CUSTOM_BBOX
        OCR_LANGS = cfg["OCR_LANGS"].strip() or OCR_LANGS
        OCR_WHITELIST = cfg["OCR_WHITELIST"]
        PSM_AUTOTUNE  = bool(cfg["PSM_AUTOTUNE"])
        LIVE_OCR_INTERVAL_SEC = float(cfg["LIVE_OCR_INTERVAL_SEC"])
        ECO_MAX_INTERVAL_SEC  = float(cfg["ECO_MAX_INTERVAL_SEC"])
        ECO_STEP_SEC          = float(cfg["ECO_STEP_SEC"])
        ECO_STABLE_TICKS_TO_STEP = int(cfg["ECO_STABLE_TICKS_TO_STEP"])
        AUTO_OCR_MIN_PIXELS   = int(cfg["AUTO_OCR_MIN_PIXELS"])
        DHASH_HAMMING_THRESHOLD = int(cfg["DHASH_HAMMING_THRESHOLD"])
        CAPTURE_MODE = cfg["CAPTURE_MODE"]
        CAPTURE_CUSTOM_BBOX = cfg["CAPTURE_CUSTOM_BBOX"]
        save_user_settings()
        # Apply to worker at runtime (no UI changes)
        if self._worker is not None:
            self._worker.set_eco_params(LIVE_OCR_INTERVAL_SEC, ECO_MAX_INTERVAL_SEC, ECO_STABLE_TICKS_TO_STEP)
            if CAPTURE_MODE in ("full","center"):
                bbox = capture_screen_bbox_of_widget(self.bar)
            else:
                bbox = dict(CAPTURE_CUSTOM_BBOX)
            self._worker.set_bbox(bbox)

    def attach_worker(self, worker: "OcrWorker"):
        self._worker = worker
        try:
            self._worker.set_bar_widget(self.bar)
        except Exception:
            pass

    @QtCore.pyqtSlot(bool)
    def on_live_toggled(self, on: bool):
        self._capture_allowed = bool(on)
        if on:
            # Prime context immediately when Live turns ON
            self.reset_all(True)
# Global exception hook for diagnostics
import traceback

def excepthook(exc_type, exc, tb):
    try:
        print("[Unhandled]", exc)
        traceback.print_exception(exc_type, exc, tb)
    except Exception:
        pass

sys.excepthook = excepthook

# ------------------- MAIN --------------------
def main():
    load_user_settings()
    if TESSERACT_CMD and not os.path.exists(TESSERACT_CMD):
        print(f"WARNING: Tesseract not found at path: {TESSERACT_CMD} - update TESSERACT_CMD.", file=sys.stderr)
    if not OPENAI_API_KEY:
        print("WARNING: Missing OPENAI_API_KEY - embeddings/LLM disabled.")

    app = QtWidgets.QApplication(sys.argv)

    bar = TopBar()
    bubble = AnswerBubble(bar)
    hist = HistoryWindow()
    bar.show()

    core = Core(bar, bubble, hist)

    # Worker + Thread (OCR background)
    thread = QtCore.QThread()
    bbox = capture_screen_bbox_of_widget(bar)  # bbox computed in GUI thread per settings
    worker = OcrWorker(bbox)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    # GUI -> worker
    bar.liveToggleSignal.connect(worker.set_live); bar.liveToggleSignal.connect(core.on_live_toggled)
    bar.ecoToggleSignal.connect(worker.set_eco)

    # worker -> Core/GUI
    worker.ocrText.connect(core.on_ocr_text)
    worker.toast.connect(lambda msg: print("[Toast]", msg))
    worker.toast.connect(lambda msg: core.toastSignal.emit(msg))
    worker.busy.connect(lambda b: None)

    core.attach_worker(worker)

    # Q&A and history
    bar.askSignal.connect(core.ask)
    bar.historyToggleSignal.connect(hist.toggle)
    bar.settingsRequested.connect(core.show_settings)
    bar.positionChanged.connect(lambda: worker.set_bbox(capture_screen_bbox_of_widget(bar)))

    # display toast
    toast = Toast(bar)

    # System tray tiny icon (not used for notifications)
    pm = QtGui.QPixmap(16, 16)
    pm.fill(QtGui.QColor(6, 182, 212))  # cyan tone
    icon = QtGui.QIcon(pm)
    tray = QtWidgets.QSystemTrayIcon(icon)
    tray.setToolTip("StudyGlance")
    tray.show()

    def show_toast(msg: str):
        try:
            toast.show_text(msg)
        except Exception:
            pass

    core.toastSignal.connect(show_toast)
    core.showAnswerSignal.connect(lambda html: bubble.show_answer(html, ANSWER_AUTOHIDE_MS))
    core.historySignal.connect(hist.append_html)

    # Foreground watcher: reset context when active window changes (Windows only)
    try:
        fg = ForegroundWatcher(interval_ms=350, parent=app)
        fg.changed.connect(core.on_window_changed)
        fg.start()
    except Exception:
        pass

    # log UI toggles
    bar.liveToggleSignal.connect(lambda s: (print("[UI] Live toggled:", s), show_toast(f"Live OCR: {'ON' if s else 'OFF'}")))
    bar.ecoToggleSignal.connect(lambda s: (print("[UI] Eco toggled:", s), show_toast(f"Eco: {'ON' if s else 'OFF'}")))

    # cleanup
    app.aboutToQuit.connect(worker.stop)
    app.aboutToQuit.connect(thread.quit)

    # start
    app.setQuitOnLastWindowClosed(True)
    bar.raise_(); bar.activateWindow(); bar.setFocus()
    toast.show_text("Overlay started (Ctrl+L Live, Ctrl+H History)", 1600)
    thread.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
















