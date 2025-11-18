"""Demo simulator: generates faux OCR events into a running backend.

Usage:
  python -m app.tools.demo_sim
"""
from __future__ import annotations

import time

import httpx


API = "http://127.0.0.1:8000"


def main() -> None:
    c = httpx.Client()
    sid = c.post(f"{API}/session/start").json()["session_id"]
    texts = [
        "Welcome to the StudyGlance demo",
        "Slide: API rate limit is 100 rpm",
        "Topic: Delta gating reduces CPU",
        "Summary: No images leave your machine",
    ]
    for t in texts:
        c.post(f"{API}/note", json={"session_id": sid, "text": t})
        time.sleep(0.2)
    print("Demo events created. Try /timeline and /ask.")


if __name__ == "__main__":
    main()

