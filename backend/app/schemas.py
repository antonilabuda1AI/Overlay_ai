from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class HealthOut(BaseModel):
    ok: bool
    versions: dict


class ConfigIn(BaseModel):
    langs: Optional[str] = None
    psm: Optional[int] = None
    interval_ms: Optional[int] = None
    delta: Optional[dict] = None


class SessionStartOut(BaseModel):
    session_id: str
    db_path: str


class SessionStopOut(BaseModel):
    saved: bool
    counts: dict


class LiveStartIn(BaseModel):
    monitor_id: Optional[int] = None
    bbox: Optional[List[int]] = Field(default=None, description="[left, top, width, height]")


class LiveStopOut(BaseModel):
    ok: bool = True


class TimelineItem(BaseModel):
    ts_ms: int
    bbox: List[int]
    text: str


class TimelineQuery(BaseModel):
    session_id: str
    from_ts: Optional[int] = None
    to_ts: Optional[int] = None
    limit: Optional[int] = 200


class NoteIn(BaseModel):
    session_id: str
    ts_ms: Optional[int] = None
    text: str


class AskIn(BaseModel):
    session_id: str
    question: str
    top_k: Optional[int] = 6


class AskOut(BaseModel):
    answer: str
    contexts: List[TimelineItem]


class ExportIn(BaseModel):
    session_id: str
    format: Literal["json", "md"]


class ExportOut(BaseModel):
    path: str

