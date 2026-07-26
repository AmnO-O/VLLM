# VChallenge — Logic Audit & Optimization Proposal

Repo state at time of writing: `053a5f996` (merge of the prefix-cache policy work
and BvoJa `294e8acaa`).

Contents:

1. [Logic audit](#1-logic-audit) — what I checked, what I found, what I fixed
2. [The one number that gates everything](#2-the-one-number-that-gates-everything)
3. [Optimization proposals, ranked](#3-optimization-proposals-ranked)
4. [Traps — changes that look good and are not](#4-traps--changes-that-look-good-and-are-not)
5. [Measurement protocol](#5-measurement-protocol)
6. [Deliverable: `docker-compose_v1.yml`](#6-deliverable-docker-compose_v1yml)

---

## 1. Logic audit

### 1.1 Verified correct

| Area | How it was checked | Result |
|---|---|---|
| `FreeKVCacheBlockQueue` under LRU | Differential test: extracted the class from both this tree and `git show HEAD:`, ran 400 random op sequences × 20 seeds | **Byte-for-byte identical behavior.** The default path is provably unchanged. |
| SLRU segment semantics | Targeted tests: probationary-drains-first, `popleft_n` spanning the boundary, `remove` from mid-protected, `prepend_n` targeting, `iter_blocks_after` crossing segments | Correct |
| Free-list invariants | Fuzz: 30 seeds × 500 mixed ops, asserting counter sync, no duplicated blocks, no lost blocks, segment sums == total | Holds |
| Filler skip / `extra_keys_list` sync | Code review of both loops in `cache_full_blocks` | Correct — the skip is mirrored in the second loop, so `extra_keys_list` stays length-matched to `new_hashes` when KV events are on |
| Config precedence | CLI > env > default, plus rejection of malformed values | Correct |
| Merge with `294e8acaa` | Verified both change sets survive in `sched/scheduler.py`; no conflict markers; compiles | Clean |

The `remove()` operation deserves a note: it needs **no** knowledge of which
segment a block is in, because the list is intrusive and unlinking goes through
the block's own `prev`/`next`. That is why SLRU here costs **zero bytes per
block** — segment membership is implied by which list a block is threaded into.

### 1.2 Bug found and fixed during this audit

**`_maybe_log_stats` was only reachable from the eviction path.** It was called
inside `_maybe_evict_cached_block`, which means that if the run produced *zero*
evictions — the single most likely outcome, and precisely the answer the Phase-0
gate exists to obtain — the summary would never have printed at all. The gate
would have looked broken rather than answered.

Fixed by also driving the periodic summary from the allocation path in
`get_new_blocks`. The gate now reports regardless of outcome.

### 1.3 Known gaps — deliberate, documented

| Gap | Impact today | If it changes |
|---|---|---|
| `cache_partial_block` does not honor `--filler-depth-tokens` | **None.** It has no production caller in this tree — only `tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py` exercises it. Gating it would break those tests. | If upstream wires it into the scheduler, deep filler blocks would start receiving partial hashes, partly defeating the skip. Re-gate then. |
| `reset_prefix_cache()` leaves blocks in the protected segment | Cosmetic. Those blocks are unhashed after reset, so eviction order among them is irrelevant; it only biases which blocks get allocated first. | Nothing |
| `log_policy_stats()` is a manual hook, never called internally | None — the periodic path covers normal use | Call it explicitly if you want a summary at a specific moment |
| The `bool \| None` CLI flags (`--enable-skip-filler-hashing`, `--prefix-cache-stats`) have not been exercised through argparse | Unverified | **The environment variables are the guaranteed path.** If a CLI flag misbehaves, use `VLLM_SKIP_FILLER_HASHING` / `VLLM_PREFIX_CACHE_STATS`. |
| No pytest run anywhere | Unverified | No torch on the dev box. See §5. |

### 1.4 The one behavioral risk you own

`--enable-skip-filler-hashing` **defaults to `true`**, per `plan.md` §1. That
stops caching *any* prefix deeper than 2000 tokens, for every workload this build
serves — not just this trace. It is correct for the grading trace, whose reusable
content all lives below token 2000. It is wrong for anything with a deeper shared
prefix.

Concretely, expect existing prefix-caching tests that use long sequences to fail.
If they do, `VLLM_SKIP_FILLER_HASHING=0` should turn them green; that confirms
the default is the cause rather than a defect.

---

## 2. The one number that gates everything

Before tuning eviction, find out whether eviction ever happens.

At startup vLLM logs (`kv_cache_utils.py:2256`):

```
GPU KV cache size: N tokens
Maximum concurrency for 4800 tokens per request: X.XXx
```

Read `X`. The trace needs roughly **70 concurrent conversations × ~4200 tokens**.

* **X >> 70** → the KV cache never fills. Evictions are zero. **No eviction
  policy — LRU, SLRU, or anything else — can change your score.** Skip §3 Tier 3
  entirely and spend the effort on Tier 1.
* **X ≈ 70 or below** → real pressure exists, SLRU is worth measuring.

My expectation is strongly the former: an H200 at `gpu-memory-utilization=0.95`
gives ~132 GB of fp8 KV cache, and LFM2.5-1.2B is a Liquid hybrid whose
conv/SSM-dominated architecture has very few full-attention layers and therefore
unusually small KV-bytes-per-token. That is a prediction from architecture, not a
measurement — the log line settles it in one line of output.

The instrumentation confirms it empirically:

```yaml
environment:
  - VLLM_PREFIX_CACHE_STATS=1
  - VLLM_LOGGING_LEVEL=INFO
```

Look for `prefix-cache stats: evictions=0 ...`. If `evictions` stays 0 across a
full replay, that is the answer.

---

## 3. Optimization proposals, ranked

Ranked by *expected value on the actual bottleneck*, which is **CPU scheduling on
3 cores**, not GPU throughput. An H200 serving a 1.2 B model is nowhere near
compute-bound; the Python scheduler loop is.

### Tier 1 — CPU wins, high confidence

**T1.1 — `--prefix-caching-hash-algo=xxhash`. Do this first.**

The default is `sha256`, which per its own docstring "uses Pickle for object
serialization before hashing". Every full block gets pickled and SHA-256'd. At
`block-size=32`, a 4000-token prompt is **125 blocks**, so a 420-request replay
performs **~52,500 pickle+SHA256 operations**, all inside the GIL-bound scheduler
process.

xxhash is a non-cryptographic 128-bit hash, typically several times faster. Your
teammate's `294e8acaa` added `xxhash` to `requirements/common.txt`, so the
dependency is now present and this costs one flag.

Security caveat worth stating plainly: xxhash raises collision risk relative to
SHA-256, which matters in multi-tenant serving. For a single-tenant benchmark it
is the right trade.

**T1.2 — `--enable-skip-filler-hashing` (already default `true`).**

Removes the hash-map insert for the ~63 write-once blocks per request — roughly
**20,800 pointless map inserts plus their matching removals** across a graded
run, all in the scheduler's critical section.

Note precisely what this does *not* do: block hashes are computed eagerly at
request construction in `get_request_block_hasher`, and truncating that would
break `resolve_block_hashes` indexing. So this saves **map churn, not hash
computation**. T1.1 is what saves hash computation, which is why it ranks higher.

**T1.3 — try `--block-size=64`.**

Block count per request halves from 125 → 63. That halves per-block hashing, map
inserts, and free-list link operations — compounding with T1.1 and T1.2. The cost
is coarser cache granularity: a 1000-token prefix covers 15.6 blocks instead of
31.2, so prefix-hit boundaries get blunter.

Verify your attention backend accepts 64 before relying on it. Treat as an A/B,
not a default — which is why `docker-compose_v1.yml` ships 32 with 64 documented.

### Tier 2 — right-size the KV cache

**T2.1 — stop allocating KV cache you cannot use.**

`BlockPool.__init__` creates **one Python `KVCacheBlock` object per block** and
links them all into the free list. If the cache is provisioned for millions of
blocks while the workload's working set is ~9,200 blocks (294 K tokens ÷ 32),
you pay for millions of objects in startup time and resident memory.

Good news, and a correction worth making explicitly: this is **not** a
steady-state GC problem. `EngineCore.__init__` builds the `Scheduler` at
`core.py:150` and calls `freeze_gc_heap()` at `core.py:232`, so those block
objects are frozen out of GC before serving begins. The cost is startup latency
and RSS, not per-step collection overhead.

So: worth doing for startup time, but do not expect it to move TTFT. Use
`--kv-cache-memory-bytes` or lower `--gpu-memory-utilization` once the log line
in §2 tells you how over-provisioned you actually are.

### Tier 3 — eviction policy (gated on §2)

**T3.1 — `--prefix-cache-policy=slru`.** Only if `evictions > 0`. The
implementation is built, tested, and zero-cost when inactive, but if nothing is
ever evicted it changes nothing. Leave at `lru` until the gate says otherwise.

---

## 4. Traps — changes that look good and are not

**4.1 — `--always-prioritize-decode` (from `294e8acaa`) probably hurts your metric.**

Read what it actually does:

```python
if self.scheduler_config.always_prioritize_decode:
    defer_prefills = throttle_prefills and has_decode_reqs
else:
    defer_prefills = (throttle_prefills and not self.prefill_capacity_bound) and has_decode_reqs
```

Enabling it **drops the `not prefill_capacity_bound` guard**, so prefills are
deferred strictly more often. Deferred prefill means a newly-arrived request
waits longer for its first token. That trades **TTFT for TPOT**.

Your stated success criteria are TTFT p50 < 50 ms and TTFT p95 reduction. This
flag pushes the wrong way on both. It is genuinely useful for a decode-latency-
bound interactive workload — but that is not what the grading trace measures,
where every request is a 4000-token prefill producing only 200 tokens.

`docker-compose_v1.yml` therefore leaves it **off**, with the A/B documented. Do
not stack it with other changes; measure it alone.

**4.2 — `--stream-interval > 1` will make your measured TTFT worse.**

It looks like an obvious CPU win: 200 output tokens × 330 requests = 66,000 SSE
chunks, and raising the interval to 4 cuts that fourfold. But the first chunk
would then contain 4 tokens instead of 1, so it is emitted **later** — and TTFT
is measured at the first token. You would trade your headline metric for
back-end CPU. Keep it at 1.

**4.3 — Do not stack ablations.** With Poisson arrivals over 303 s on 3 cores,
run-to-run variance will likely exceed the effect size of any single change. One
variable at a time, ≥3 runs each.

---

## 5. Measurement protocol

**Step 0 — run the tests.** Nothing here has been through pytest; there is no
torch on the dev box. On the GPU machine:

```bash
pytest tests/v1/core/test_prefix_cache_policy.py \
       tests/v1/core/test_kv_cache_utils.py \
       tests/v1/core/test_prefix_caching.py -x -q
```

If the third fails, retry with `VLLM_SKIP_FILLER_HASHING=0`. Green on retry
confirms §1.4 rather than a defect.

**Step 1 — read the gate.** Start the server, read `Maximum concurrency` (§2).
This decides whether Tier 3 is worth any effort.

**Step 2 — ablate, one variable at a time.** Run ablation A first: it bounds
every other number. If disabling prefix caching entirely barely moves TTFT, the
prefix-cache axis is not where your score is, and you should redirect to Tier 1
and scheduler tuning.

| # | Configuration | Isolates |
|---|---|---|
| A | drop `--enable-prefix-caching` | **Upper bound on all prefix-cache work** |
| B | baseline (sha256, no filler skip, lru) | reference |
| C | B + `--prefix-caching-hash-algo=xxhash` | T1.1 |
| D | C + `--enable-skip-filler-hashing` | T1.2 |
| E | D + `--block-size=64` | T1.3 |
| F | D + `--prefix-cache-policy=slru` | T3.1 (only if gate passed) |
| G | D + `--always-prioritize-decode` | §4.1 — expect regression |

**Step 3 — replay.**

```bash
pip install aiohttp
python VChallenge/replay_trace.py \
    --base-url http://localhost:8000 \
    --model LFM2.5-1.2B-Instruct \
    --trace ../input_part2/trace_grading_public.jsonl \
    --label "C-xxhash" --json-out results_C.json
```

The harness excludes the 90 warmup rows (conversations 0–14) from all latency
statistics — including them washes out every policy difference, because warmup is
dominated by cold-start prefill. It exits non-zero if any graded request fails.

**Success criteria, in priority order:**

1. `failed_count == 0` — a hard gate. A free-list accounting bug surfaces as
   `ValueError("No free blocks available")` and fails requests, which costs more
   than any latency gain earns.
2. TTFT p50 < 50 ms on graded requests
3. TTFT p95 reduction
4. ERS maximization

---

## 6. Deliverable: `docker-compose_v1.yml`

Shipped alongside this file. Its choices, and why:

| Setting | Value | Reason |
|---|---|---|
| `--prefix-caching-hash-algo` | `xxhash` | T1.1 — the largest CPU win available |
| `--enable-skip-filler-hashing` | on | T1.2 |
| `--filler-depth-tokens` | `2000` | Trace layout: shared 1000 + conv 1000 reusable, remainder is filler |
| `--prefix-cache-policy` | `lru` | Gated on §2; SLRU is built but changes nothing at zero evictions |
| `--slru-protected-tokens` | `1000` | Pre-set so flipping to `slru` is a one-word edit |
| `--always-prioritize-decode` | **off** | §4.1 — trades away the metric being scored |
| `--stream-interval` | default (1) | §4.2 |
| `--block-size` | `32` | 64 is a documented A/B, not a default |
| `--gpu-memory-utilization` | `0.95` | Unchanged pending the §2 reading |
| `VLLM_PREFIX_CACHE_STATS` | off | Zero hot-path cost for the submission; a commented measurement block flips it on |
| `VLLM_LOGGING_LEVEL` | `ERROR` | Less I/O on 3 cores. Measurement profile uses `INFO`. |

Everything tunable is a CLI flag *and* an environment variable, resolved
**CLI > env > default**, so the whole matrix in §5 is reachable from compose
without touching source.

**Deployment caveat, unchanged and still open:** these flags exist only in a
build of this tree. `vllm/vllm-openai:v0.22.1` will reject them with
"unrecognized arguments". The compose file carries a `build:` block for that
reason; the volume-mount alternative is only valid against an image whose vLLM
version matches this checkout, which `v0.22.1` does not.
