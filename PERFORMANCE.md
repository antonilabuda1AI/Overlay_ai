Performance Guide â€” OverlayAI
===============================

Overview
- The backend uses delta-diff gating, ROI merge, a small OCR pool, and batched SQLite writes to keep CPU/memory low and latency predictable.
- Instrumentation logs per-tick CSV metrics to `~/.OverlayAI/logs/perf-<date>.csv`.

Instrumentation (CSV fields)
- ts_ms: tick timestamp
- capture_ms, delta_ms, ocr_ms, db_ms, total_ms: stage timings
- rois, skipped_rois, ocr_hits, cache_hits, dropped_frames: counters per tick

Profiling harness
- Capture loop: `python scripts/profile_capture.py --ticks 1000 --interval-ms 200 --max-rois 5 [--synthetic]`
- API loop: `python scripts/profile_api.py --qps 10 --duration 60`
- Line profiler: `./scripts/line_profile.sh backend/app/ocr/delta.py`
- Memory trace: `./scripts/memtrace.sh backend/app/main.py`
- Compare CSVs: `python scripts/compare_perf.py before.csv after.csv`

Tuning flags (env or /config)
- OCR_INTERVAL_MS, OCR_INTERVAL_MS_MIN/MAX: capture cadence
- DELTA_THR (DELTA_GLOBAL_DIFF_THR): gate sensitivity
- DELTA_MAX_ROIS, DELTA_MAX_ROIS_PER_TICK: OCR work budget caps
- ROI_DHASH_COOLDOWN_MS: duplicate ROI suppression window
- OCR_POOL_SIZE: threads for ROI OCR (1â€“3 recommended)
- DROP_POLICY: backlog handling for ROI queues (newest|oldest|coalesce)
- PSM_AUTOTUNE: choose PSM=7 for narrow boxes, else PSM=6
- WHITELIST_REGEX: optional `tessedit_char_whitelist` for speed
- DB_WRITE_BATCH_MS / DB_WRITE_BATCH_ROWS: SQLite batching
- EMBED_BATCH_SIZE / SEARCH_TIMEOUT_MS: RAG throughput guards

Known tradeâ€‘offs
- Higher DELTA_THR reduces OCR freq but may miss subtle changes.
- Larger OCR_POOL_SIZE increases concurrency but risks CPU spikes.
- Aggressive DB batching reduces sync latency but risks slightly delayed timeline persistence.
- Whitelist speeds OCR for constrained character sets but harms generality if misconfigured.

Targets
- Idle (unchanged): median < 120ms; OCR=0, DB writes minimal
- Small changes (1â€“3 ROIs): median < 350ms; OCR/ROI < 180ms
- Dropped frames < 10% under bursts with defaults
- DB flush p95 < 10ms (batched)
- Memory stable (< 5% over 10k ticks)

Hotspots
- `ocr/delta.extract_rois`: contour + merge â€” memory friendly via OpenCV; keep kernel cached
- `ocr/tesseract.batch_ocr`: PSM autotune + optional whitelist
- `live_worker`: single mss instance, debounce if OCR busy, small thread pool
- `storage/session_db`: WAL + batched writes with prepared statement

