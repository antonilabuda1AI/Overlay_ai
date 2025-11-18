from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import SETTINGS
from .routes import router


def setup_logging() -> None:
    log_dir = SETTINGS.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "backend.log"
    handler = logging.handlers.RotatingFileHandler(str(logfile), maxBytes=1_000_000, backupCount=3)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), handler])


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


def create_app() -> FastAPI:
    # Enable uvloop where available (Linux/macOS)
    try:
        import sys
        if sys.platform != "win32":
            import uvloop  # type: ignore
            uvloop.install()
    except Exception:
        pass
    app = FastAPI(title="StudyGlance API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://localhost:5173", "http://127.0.0.1"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
