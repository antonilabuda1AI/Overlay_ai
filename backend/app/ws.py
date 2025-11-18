from __future__ import annotations

from typing import Any, Dict, List

from fastapi import WebSocket


class LiveHub:
    def __init__(self) -> None:
        self._clients: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._clients:
            self._clients.remove(ws)

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        dead: List[WebSocket] = []
        for c in self._clients:
            try:
                await c.send_json(payload)
            except Exception:
                dead.append(c)
        for d in dead:
            self.disconnect(d)

LIVE_HUB = LiveHub()

