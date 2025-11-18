#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--qps", type=int, default=10)
    ap.add_argument("--duration", type=int, default=30)
    args = ap.parse_args()

    c = httpx.Client()
    sid = c.post(f"{args.base}/session/start").json()["session_id"]
    t_end = time.time() + args.duration
    interval = 1.0 / max(1, args.qps)
    print("endpoint,latency_ms,status")
    while time.time() < t_end:
        for path, method, payload in [
            ("/live/start", "post", {}),
            ("/timeline", "get", {"session_id": sid}),
            ("/ask", "post", {"session_id": sid, "question": "summarize"}),
        ]:
            t0 = time.perf_counter_ns()
            try:
                if method == "get":
                    r = c.get(f"{args.base}{path}", params=payload)
                else:
                    r = c.post(f"{args.base}{path}", json=payload)
                ms = (time.perf_counter_ns() - t0) / 1e6
                print(f"{path},{ms:.2f},{r.status_code}")
            except Exception:
                ms = (time.perf_counter_ns() - t0) / 1e6
                print(f"{path},{ms:.2f},ERR")
            time.sleep(interval)
    c.post(f"{args.base}/session/stop", params={"session_id": sid})


if __name__ == "__main__":
    main()

