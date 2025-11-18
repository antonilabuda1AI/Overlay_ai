from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


def export_json(path: Path, events: List[Tuple[int, list, str]]) -> Path:
    out = [
        {"ts_ms": ts, "bbox": bbox, "text": text}
        for (ts, bbox, text) in events
    ]
    out_path = path.with_suffix(".json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def export_markdown(path: Path, events: List[Tuple[int, list, str]]) -> Path:
    lines = ["# StudyGlance Export\n"]
    for ts, bbox, text in events:
        lines.append(f"- {ts} {bbox} — {text}")
    out_path = path.with_suffix(".md")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path

