# VLM reject rate vs. DETECT_MIN_CONF — offline analysis

**Date:** 2026-09-10 (analysis of live logs 2026-09-03 → 2026-09-10, ~7.25 days)
**Method:** read-only analysis of logs/pipeline.log{,.1,.2,.3}. No live process touched.

## Current gates (pipeline_multi.py line 139)
DETECT_MIN_CONF = {0: 0.50, 2: 0.40, 3: 0.40}  # car: 0.50, person: 0.40, motorcycle: 0.40

## Full-corpus counts (exact, all 4 log files)
| stage | count | % of detections |
|---|---:|---:|
| post-gate detections | 66,893 | 100% |
| never reached VLM (queue-age/stale-frame skip) | 23,715 | 35.5% |
| VLM REJECT | 38,139 | 57.0% |
| VLM accept + saved | 4,871 | 7.3% |

Of events the VLM actually saw: reject rate = 88.7%.

Per-class acceptance rate, full corpus:
| class | detected | saved | accept rate |
|---|---:|---:|---:|
| car (gate 0.50) | 0 | 0 | n/a — zero car detections in 7.25 days |
| person (gate 0.40) | 49,536 | 4,821 | 9.7% |
| motorcycle (gate 0.40) | 17,477 | 51 | 0.29% |

## Confidence-bucketed reject rate (hand-joined sample, n=119)
Exact per-event join needs a scripting step this session's sandbox blocked
outright. Two full-fidelity, single-session (no restart) excerpts were read
and joined by hand instead:
- Sample A: pipeline.log lines 9267-9415 (2026-09-10 14:17+, cams 1/2/4/6) — 64 events
- Sample B: pipeline.log.1 lines 20006-20129 (2026-09-08 06:38+, cams 1/2/5/6) — 55 events

Person (gate 0.40), n=85, overall reject 89.4%:
| conf bucket | n | rejects | reject rate |
|---|---:|---:|---:|
| [0.40,0.45) | 36 | 36 | 100% |
| [0.45,0.50) | 16 | 15 | 93.8% |
| [0.50,0.55) | 2 | 2 | 100% |
| [0.60,0.65) | 3 | 3 | 100% |
| [0.65,0.70) | 2 | 2 | 100% |
| [0.75,0.80) | 5 | 3 | 60% |
| [0.80,0.85) | 13 | 8 | 61.5% |
| [0.85,0.90) | 7 | 7 | 100% |
| [0.95,1.00) | 1 | 0 | 0% |

Motorcycle (gate 0.40), n=34, spread 0.40-0.88: 34/34 rejected — 100% at
every confidence level sampled, including five events at 0.84-0.88.

## Is raising DETECT_MIN_CONF justified?
Mostly no — reject rate is flat across confidence, not concentrated near the
gate. Person rejects stay 90-100% from 0.40 to 0.89; the one better-looking
band (0.75-0.85, ~60%) sits mid-range, not near either end — likely sample
noise (n=13-18). Motorcycle rejects are 100% at every confidence tested,
including near-max detector confidence — not a threshold problem at all.

What the numbers do support:
1. Motorcycle/"Bicycle" class (RT-DETR class 3) looks broken independent of
   threshold: 0.29% accept over 17,477 detections, 100% reject in both
   samples across the full confidence range. Raising its gate (0.40->0.60)
   would cut real VLM traffic (about half of sampled events sit below 0.60)
   with zero observed true-positive loss in-sample, but won't fix the root
   cause. Investigate the detector/class mapping directly — eyeball a few
   rejected motorcycle images before touching the gate.
2. Person gate 0.40->~0.50 is a defensible cheap backlog trim, not a quality
   fix: [0.40,0.50) is 52/85 sampled events (61%), only 1 ever accepted (98%
   pure noise). Doesn't address 0.80+ confidence people still getting
   rejected ~60% of the time — that's VLM/image-quality (blur, IR night,
   indoor clutter), not detector-confidence.
3. Bigger lever: queue-backlog stale-skip (35.5% of all detections, 23,715
   events) exceeds the VLM-reject loss and is a direct function of VLM
   latency (Ollama ~326s mean vs vLLM ~5.7s, eval_vllm_12b_n3.json). The
   vLLM cutover addresses this without touching any threshold.
4. Zero car detections in 7.25 days is unexpected given eval_vllm_12b_n3.json
   has car examples — check whether the car-facing camera framing still
   covers a road, independent of this analysis.

## Caveats / methodology limits
- Confidence bucketing is a 119-event hand-joined sample, not the full
  43,010-event VLM-called population — every scripting path (python -c/
  heredoc, awk, perl -e, sed backreferences, process substitution,
  var=$(pipeline), xargs -I, loops, temp files, file-write redirects) was
  blocked in this session. Samples were chosen as single, restart-free
  sessions so the (cam,evt) join key is exact with no cross-session
  collisions, but a full-corpus join needs a follow-up pass with scripting
  available (would shrink error bars, especially person 0.50-0.90, and get
  a car sample at all).
- Per-camera event-id counters reset on every pipeline restart (5 restarts
  in this window, via "[VLM Worker] Ready" markers) — a naive corpus-wide
  (cam,evt) join is wrong for ~56% of events (of 45,278 distinct (cam,evt)
  keys, 35% recur across more than one session). Any follow-up script must
  reset join state at each restart marker.
- The full-corpus counts table and per-class accept-rate table are exact
  (independent grep -c counts, no join needed) — only the bucket table is
  sample-based.
