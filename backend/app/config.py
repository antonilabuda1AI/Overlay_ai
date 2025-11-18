from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class DeltaConfig:
    enabled: bool = True
    thr: int = 12
    min_area: int = 120
    max_rois: int = 12
    max_rois_per_tick: int = 6


@dataclass
class Settings:
    openai_api_key: Optional[str]
    data_dir: Path
    ocr_langs: str = "eng+pol"
    psm: int = 6
    interval_ms: int = 250
    interval_ms_min: int = 100
    interval_ms_max: int = 1000
    delta: DeltaConfig = field(default_factory=DeltaConfig)
    roi_cooldown_ms: int = 600
    ocr_pool_size: int = 2
    drop_policy: str = "newest"  # newest|oldest|coalesce
    psm_autotune: bool = True
    whitelist_regex: str = ""
    db_write_batch_ms: int = 250
    db_write_batch_rows: int = 50
    embed_batch_size: int = 64
    search_timeout_ms: int = 800
    # Force local LLM/embeddings (offline fallback even if key is present)
    llm_force_local: bool = False

    def ensure_dirs(self) -> None:
        (self.data_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "sessions").mkdir(parents=True, exist_ok=True)


def _bool(s: str | None, default: bool) -> bool:
    if s is None:
        return default
    return s.lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    load_dotenv(override=False)
    data_dir = Path(os.getenv("DATA_DIR", str(Path.home() / ".studyglance"))).expanduser()
    delta = DeltaConfig(
        enabled=_bool(os.getenv("DELTA_ENABLED"), True),
        thr=int(os.getenv("DELTA_THR", "12")),
        min_area=int(os.getenv("DELTA_MIN_AREA", "120")),
        max_rois=int(os.getenv("DELTA_MAX_ROIS", "12")),
        max_rois_per_tick=int(os.getenv("DELTA_MAX_ROIS_PER_TICK", "6")),
    )
    settings = Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        data_dir=data_dir,
        ocr_langs=os.getenv("OCR_LANGS", "eng+pol"),
        psm=int(os.getenv("PSM", "6")),
        interval_ms=int(os.getenv("OCR_INTERVAL_MS", "250")),
        interval_ms_min=int(os.getenv("OCR_INTERVAL_MS_MIN", "100")),
        interval_ms_max=int(os.getenv("OCR_INTERVAL_MS_MAX", "1000")),
        delta=delta,
        roi_cooldown_ms=int(os.getenv("ROI_DHASH_COOLDOWN_MS", "600")),
        ocr_pool_size=int(os.getenv("OCR_POOL_SIZE", "2")),
        drop_policy=os.getenv("DROP_POLICY", "newest"),
        psm_autotune=_bool(os.getenv("PSM_AUTOTUNE"), True),
        whitelist_regex=os.getenv("WHITELIST_REGEX", ""),
        db_write_batch_ms=int(os.getenv("DB_WRITE_BATCH_MS", "250")),
        db_write_batch_rows=int(os.getenv("DB_WRITE_BATCH_ROWS", "50")),
        embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "64")),
        search_timeout_ms=int(os.getenv("SEARCH_TIMEOUT_MS", "800")),
        llm_force_local=_bool(os.getenv("LLM_FORCE_LOCAL"), False),
    )
    settings.ensure_dirs()
    return settings


SETTINGS = load_settings()
