#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv


def load(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def pct(a: float, b: float) -> float:
    if a == 0:
        return 0.0
    return (b - a) / a * 100.0


def summarize(rows):
    def col(name):
        vals = [float(x[name]) for x in rows]
        vals.sort()
        n = len(vals)
        med = vals[n//2] if n else 0.0
        p95 = vals[int(n*0.95)] if n else 0.0
        return med, p95
    res = {}
    for k in ["capture_ms","delta_ms","ocr_ms","db_ms","total_ms"]:
        res[k] = col(k)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    args = ap.parse_args()
    b = summarize(load(args.before))
    a = summarize(load(args.after))
    print("metric,median_before,median_after,delta%,p95_before,p95_after,delta%")
    for k in ["capture_ms","delta_ms","ocr_ms","db_ms","total_ms"]:
        med_b, p95_b = b[k]
        med_a, p95_a = a[k]
        print(f"{k},{med_b:.2f},{med_a:.2f},{pct(med_b,med_a):.1f}%,{p95_b:.2f},{p95_a:.2f},{pct(p95_b,p95_a):.1f}%")


if __name__ == "__main__":
    main()

