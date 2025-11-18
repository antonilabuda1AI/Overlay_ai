from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket
from fastapi.responses import JSONResponse

from .config import SETTINGS
from .llm.openai_client import OpenAIClient
from .rag.index import Doc, VectorIndex
from .rag.select import mmr
from .schemas import (
    AskIn,
    AskOut,
    ConfigIn,
    ExportIn,
    ExportOut,
    HealthOut,
    LiveStartIn,
    LiveStopOut,
    NoteIn,
    SessionStartOut,
    SessionStopOut,
    TimelineItem,
)
from .storage.export import export_json, export_markdown
from .storage.session_db import SessionDB
from .ws import LIVE_HUB
from .live_worker import LIVE_OBSERVER


router = APIRouter()


# In-memory session state (lightweight)
SESSIONS: dict[str, dict] = {}


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    import platform
    import fastapi

    return HealthOut(ok=True, versions={"python": platform.python_version(), "fastapi": fastapi.__version__})


ACTIVE_SESSION: Optional[str] = None


@router.post("/session/start", response_model=SessionStartOut)
def session_start() -> SessionStartOut:
    sid = uuid.uuid4().hex[:8]
    db_path = SETTINGS.data_dir / "sessions" / f"{sid}.sqlite3"
    db = SessionDB.open(db_path)
    SESSIONS[sid] = {
        "db": db,
        "index": None,  # filled when we get first OCR/note
        "docs": [],
    }
    global ACTIVE_SESSION
    ACTIVE_SESSION = sid
    return SessionStartOut(session_id=sid, db_path=str(db_path))


@router.post("/session/stop", response_model=SessionStopOut)
def session_stop(session_id: str) -> SessionStopOut:
    st = SESSIONS.get(session_id)
    if not st:
        raise HTTPException(404, "session not found")
    db: SessionDB = st["db"]
    cnt = db.conn.execute("SELECT COUNT(1) FROM events").fetchone()[0]
    db.close()
    del SESSIONS[session_id]
    return SessionStopOut(saved=True, counts={"events": int(cnt)})


@router.post("/config")
def update_config(cfg: ConfigIn) -> JSONResponse:
    if cfg.langs:
        SETTINGS.ocr_langs = cfg.langs
    if cfg.psm is not None:
        SETTINGS.psm = cfg.psm
    if cfg.interval_ms is not None:
        SETTINGS.interval_ms = cfg.interval_ms
    if cfg.delta is not None:
        d = SETTINGS.delta
        d.enabled = bool(cfg.delta.get("enabled", d.enabled))
        d.thr = int(cfg.delta.get("thr", d.thr))
        d.min_area = int(cfg.delta.get("min_area", d.min_area))
        d.max_rois = int(cfg.delta.get("max_rois", d.max_rois))
        d.max_rois_per_tick = int(cfg.delta.get("max_rois_per_tick", d.max_rois_per_tick))
    # Optional extras (if supplied in payload)
    if cfg.delta is not None and "cooldown_ms" in cfg.delta:
        SETTINGS.roi_cooldown_ms = int(cfg.delta["cooldown_ms"])  # type: ignore[index]
    return JSONResponse({
        "langs": SETTINGS.ocr_langs,
        "psm": SETTINGS.psm,
        "interval_ms": SETTINGS.interval_ms,
        "delta": SETTINGS.delta.__dict__,
        "roi_cooldown_ms": SETTINGS.roi_cooldown_ms,
        "ocr_pool_size": SETTINGS.ocr_pool_size,
        "drop_policy": SETTINGS.drop_policy,
        "psm_autotune": SETTINGS.psm_autotune,
        "whitelist_regex": SETTINGS.whitelist_regex,
        "db_write_batch_ms": SETTINGS.db_write_batch_ms,
        "db_write_batch_rows": SETTINGS.db_write_batch_rows,
        "embed_batch_size": SETTINGS.embed_batch_size,
        "search_timeout_ms": SETTINGS.search_timeout_ms,
    })


@router.post("/live/start")
def live_start(body: LiveStartIn) -> JSONResponse:
    if ACTIVE_SESSION is None or ACTIVE_SESSION not in SESSIONS:
        raise HTTPException(400, "no active session; call /session/start first")
    db: SessionDB = SESSIONS[ACTIVE_SESSION]["db"]
    LIVE_OBSERVER.start(db=db, monitor_id=body.monitor_id, bbox=body.bbox)
    return JSONResponse({
        "ok": True,
        "config": {
            "interval_ms": SETTINGS.interval_ms,
            "psm": SETTINGS.psm,
            "langs": SETTINGS.ocr_langs,
            "delta": SETTINGS.delta.__dict__,
        },
        "session_id": ACTIVE_SESSION,
    })


@router.post("/live/stop", response_model=LiveStopOut)
def live_stop() -> LiveStopOut:
    LIVE_OBSERVER.stop()
    return LiveStopOut(ok=True)


@router.get("/timeline", response_model=list[TimelineItem])
def timeline(session_id: str, from_ts: Optional[int] = None, to_ts: Optional[int] = None, limit: int = 200):
    st = SESSIONS.get(session_id)
    if not st:
        raise HTTPException(404, "session not found")
    db: SessionDB = st["db"]
    rows = db.query(session_id, from_ts, to_ts, limit)
    return [TimelineItem(ts_ms=ts, bbox=bbox, text=text) for ts, bbox, text in rows]


@router.post("/note")
async def note(body: NoteIn):
    st = SESSIONS.get(body.session_id)
    if not st:
        raise HTTPException(404, "session not found")
    ts = body.ts_ms or int(time.time() * 1000)
    st["db"].append_event(ts, [0, 0, 0, 0], body.text, source="note")
    await LIVE_HUB.broadcast({"ts_ms": ts, "type": "note", "text": body.text, "bbox": [0, 0, 0, 0]})
    return {"ok": True}


@router.post("/ask", response_model=AskOut)
def ask(body: AskIn) -> AskOut:
    st = SESSIONS.get(body.session_id)
    if not st:
        raise HTTPException(404, "session not found")
    # Build/maintain index lazily
    db = st["db"]
    rows = db.query(body.session_id, None, None, 10000)
    texts = [t for _, _, t in rows]
    ts_list = [ts for ts, _, _ in rows]
    if not texts:
        return AskOut(answer="No context yet.", contexts=[])

    client = OpenAIClient()
    emb = client.embed(texts)
    idx = VectorIndex(emb.shape[1])
    idx.add(emb, [Doc(ts, t) for ts, t in zip(ts_list, texts)])
    qv = client.embed([body.question])[0]
    sims = idx.search(qv, top_k=max(6, body.top_k or 6))
    # MMR for diversity
    cand_vecs = [client.embed([d.text])[0] for d, _ in sims]
    picks = mmr(qv, cand_vecs, top_k=body.top_k or 6)
    contexts = [(sims[i][0].ts_ms, sims[i][0].text) for i in picks]
    answer = client.chat(body.question, [c[1] for c in contexts])
    items = [TimelineItem(ts_ms=ts, bbox=[0, 0, 0, 0], text=txt) for ts, txt in contexts]
    return AskOut(answer=answer, contexts=items)


@router.post("/export", response_model=ExportOut)
def export_(body: ExportIn) -> ExportOut:
    st = SESSIONS.get(body.session_id)
    if not st:
        raise HTTPException(404, "session not found")
    db: SessionDB = st["db"]
    rows = db.query(body.session_id, None, None, 100000)
    p = db.path
    if body.format == "json":
        out = export_json(p, rows)
    else:
        out = export_markdown(p, rows)
    return ExportOut(path=str(out))


@router.websocket("/live")
async def live_ws(ws: WebSocket):
    await LIVE_HUB.connect(ws)
    try:
        while True:
            # We mainly broadcast from server to clients; just keep alive
            await ws.receive_text()
    except Exception:
        pass
    finally:
        LIVE_HUB.disconnect(ws)
