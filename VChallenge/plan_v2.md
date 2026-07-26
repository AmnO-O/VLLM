# Plan: further vLLM changes for a higher VChallenge score

*Status: **implemented**. See the "Outcome" section at the end.*

## Context

You asked whether I'd edit vLLM's code further to raise the score. Short answer:
**barely.** One code change is clearly worth making. Everything else that's left
is configuration, one factual correction to `propose.md`, and one bug in my own
replay harness.

The reason for restraint is the scoring rule. `failed_count == 0` is a hard gate,
and a free-list or scheduler accounting bug surfaces as
`ValueError("No free blocks available")` → failed requests → a score worse than
doing nothing. Invasive edits to the scheduler are high-variance bets against a
metric that is probably not GPU-bound and possibly not even cache-bound. The
remaining headroom is mostly in cheap CPU wins, not clever algorithms.

**I also found a real error in what I told you earlier.** See §1 — it flips a
"don't do this" into a free win, so it comes first.

---

## 1. Correct `propose.md` §4.2 — I was wrong about `--stream-interval`

I claimed raising `--stream-interval` above 1 would inflate measured TTFT because
the first chunk would carry N tokens and be emitted later. **That is false.**

`vllm/v1/engine/output_processor.py:287-300` already special-cases it:

```python
if self.stream_interval > 1:
    # Send output request only when
    # 1. It has finished, or
    # 2. It is the first token, or
    # 3. It has reached the stream interval number of tokens
    if not (finished
            or self.sent_tokens_offset == 0          # <-- first token, always sent
            or self.detokenizer.num_output_tokens() - self.sent_tokens_offset
               >= self.stream_interval):
        return None
```

The first token is emitted immediately regardless of the interval. So
`--stream-interval` is a **free CPU win**: TTFT is untouched, and the remaining
199 tokens per request are batched.

Concretely: 330 graded requests × 200 tokens ≈ **66,000 SSE chunks**, each costing
a detokenize step, a JSON serialization and a socket write in the frontend
process. At `--stream-interval=4` that drops to ~16,500. On a 3-core box the
frontend competes with the engine-core process for CPU, so cutting frontend work
frees scheduling headroom — which is exactly where TTFT p95 is lost.

**Action:** rewrite §4.2 of `propose.md` from a trap into a Tier-1 recommendation,
and add `--stream-interval=4` to `docker-compose_v1.yml` as an ablation (not yet a
default — see the caveat in §3).

---

## 2. The one code change worth making: cap block-hash computation at filler depth

### The waste

`--enable-skip-filler-hashing` currently stops filler blocks from entering the
hash *map*, but their hashes are still **computed**. At `--block-size=32` a
4000-token prompt produces 125 block hashes, of which only the first ~62 (depth
≤ 2000) can ever be used. That is **~63 wasted hash computations per request,
~26,000 per graded run**, each pickling a 32-token list and running SHA-256
inside the GIL-bound path.

`--prefix-caching-hash-algo=xxhash` makes each of those cheaper. This change
removes half of them outright. The two compose.

### Why it's tractable

`vllm/v1/core/kv_cache_utils.py:789` `request_block_hasher` derives its resume
point purely from list length:

```python
start_token_idx = len(request.block_hashes) * hash_block_size
```

So a cap is a clean early return: once `len(block_hashes) * hash_block_size`
reaches the cap, return `[]`. `Request.update_block_hashes()`
(`vllm/v1/request.py:247`) then extends by nothing, and the guard is O(1) on every
subsequent call. The list simply stops growing at the cap.

### Files to change

| File | Change |
|---|---|
| `vllm/v1/core/kv_cache_utils.py` (`get_request_block_hasher`, ~line 775) | Accept a `max_hash_depth_tokens` parameter; early-return `[]` once the cap is reached |
| `vllm/v1/core/kv_cache_utils.py` (`resolve_block_hashes`, ~line 2254) | Ensure it tolerates a hash list shorter than the request's full block count |
| `vllm/v1/core/block_pool.py` (`cache_full_blocks`, ~line 261) | Bound the loop by `len(block_hashes)`, not `num_full_blocks` — currently `new_block_hashes[i]` would `IndexError` past the cap |
| Caller of `get_request_block_hasher` | Pass the cap, derived from the same `PrefixCachePolicyConfig` that already carries `filler_depth_tokens` |

`find_longest_cache_hit` (`vllm/v1/core/single_type_kv_cache_manager.py:431`)
needs no change — it already takes `max_length` and iterates a `BlockHashList`, so
a shorter list naturally terminates the hit walk at the cap, which is the desired
behavior.

### Reuse, not new code

The cap value is `PrefixCachePolicyConfig.filler_depth_tokens`, which already
exists in `vllm/v1/core/prefix_cache_policy.py` and is already plumbed to
`BlockPool`. Gate it on the existing `enable_skip_filler_hashing` flag so a single
switch controls both halves of the optimization, and `VLLM_SKIP_FILLER_HASHING=0`
restores stock behavior completely.

### Risk

Medium — this is the one place where an off-by-one becomes an `IndexError` in the
scheduler, i.e. failed requests. It must be covered by tests before it runs
against the grader. The mitigation is that it is fully disabled by one existing
flag.

---

## 3. Fix the replay harness's token counting

`VChallenge/replay_trace.py` does `n_tokens += 1` per SSE chunk containing text.
That is only correct at `--stream-interval=1`. At interval 4 each chunk carries ~4
tokens, so `n_tokens` undercounts ~4× and the reported TPOT becomes
*chunks*-per-second, not tokens-per-second — silently making a config look better
than it is.

**Action:** count tokens rather than chunks (request `stream_options.include_usage`
and read the usage block, or tokenize the accumulated text), so TPOT stays
comparable across stream-interval settings.

**Caveat that makes this matter:** I do not know whether the official grader
measures inter-*chunk* or inter-*token* latency. If it measures inter-chunk,
lumpy 4-token delivery could look worse to it even though total CPU dropped. So
`--stream-interval` goes in as an **ablation**, measured, not as a blind default.

---

## 4. Explicitly not doing (and why)

| Candidate | Verdict |
|---|---|
| Memoize the shared-prefix hash chain across requests | The 31 shared-prefix blocks are identical across all 420 requests, but §2 already caps total work at ~62 blocks and xxhash makes each cheap. A memo dict adds a lookup per block that may cost more than the hash it saves. Revisit only if a profile shows hashing still hot. |
| Batch SSE writes in the frontend | §1 shows vLLM already supports this via `--stream-interval` with correct first-token semantics. No code needed. |
| Custom scheduler / admission policy | High variance against `failed_count == 0` for speculative gain. The scheduler is not proven to be the bottleneck. |
| `--prefix-cache-policy=slru` | Still gated on the eviction count being non-zero. Built and tested; costs nothing while inactive. |

---

## 5. Ordering

1. **Correct `propose.md` §4.2** (§1) — documentation, zero risk, prevents you from skipping a free win
2. **Fix harness token counting** (§3) — needed *before* measuring stream-interval, or the measurement lies
3. **Implement the hash cap** (§2) — the one real code change
4. Re-run the ablation matrix with `--stream-interval` added

Steps 1–2 are prerequisites for trusting step 4's numbers.

---

## Verification

**Unit** — extend `tests/v1/core/test_prefix_cache_policy.py`:
- hash list length caps at exactly `filler_depth_tokens / hash_block_size`
- `cache_full_blocks` does not `IndexError` when full blocks exceed the hash list
- a prefix at depth ≤ cap still produces a cache hit
- `VLLM_SKIP_FILLER_HASHING=0` restores an uncapped hash list

**Regression** — the cap must not change behavior when disabled:
```bash
pytest tests/v1/core/test_prefix_cache_policy.py \
       tests/v1/core/test_kv_cache_utils.py \
       tests/v1/core/test_prefix_caching.py -x -q
```
plus the same run under `VLLM_SKIP_FILLER_HASHING=0`, which must be green
unconditionally — that is the proof the change is opt-in.

**End-to-end** — the hard gate first:
```bash
python VChallenge/replay_trace.py --base-url http://localhost:8000 \
    --model LFM2.5-1.2B-Instruct \
    --trace ../input_part2/trace_grading_public.jsonl \
    --label hashcap --json-out results_hashcap.json
```
`failed_count == 0` is non-negotiable; the harness exits non-zero otherwise. Then
compare TTFT p50/p95 against the config-only run to isolate what §2 actually
bought.

**Instrumentation cross-check** — with `VLLM_PREFIX_CACHE_STATS=1`,
`hash_inserts + hash_inserts_skipped` should fall by roughly half once the cap is
active. If it doesn't, the cap isn't taking effect.

---

## Honest summary

The largest remaining wins are **config, not code**: `--prefix-caching-hash-algo=
xxhash`, `--stream-interval=4` (per the §1 correction), and possibly
`--block-size=64`. The single code change worth the risk is the hash cap in §2,
and it is worth it mainly because it completes an optimization that is currently
half-implemented — paying to compute hashes it then deliberately throws away.

Before any of this, the gate from `propose.md` §2 still stands unanswered: read
`Maximum concurrency for 4800 tokens per request` from the startup log. It costs
one line of output and determines whether a whole tier of this work matters.


---

## Outcome

All four steps are implemented as of this commit.

| Step | Status | Where |
|---|---|---|
| §1 correct the `--stream-interval` claim | done | `propose.md` T1.4 + corrected §4.2; ablation H added; `docker-compose_v1.yml` |
| §3 harness counts tokens, not chunks | done | `replay_trace.py` — `stream_options.include_usage`, `output_tokens` vs `stream_chunks` |
| §2 hash cap | done | `kv_cache_utils.py`, `engine/core.py`, `block_pool.py` |
| tests | done | `tests/v1/core/test_prefix_cache_policy.py` (28 tests) |

### What the cap actually does

`get_request_block_hasher` takes `max_hash_depth_tokens`. `engine/core.py` derives
it from `PrefixCachePolicyConfig` and passes it **only when no KV connector is
active** — P/D and offloading consume the full block-hash list, so the cap would
corrupt them.

`cache_full_blocks` now routes both of its loops through a single
`is_uncacheable(rel_idx)` predicate. That guarantees the two loops agree (a
disagreement desyncs `extra_keys_list` from `new_hashes` under KV events) and
bounds the index independently of the depth threshold, so any drift between the
two degrades into "not cached" rather than an `IndexError` in the scheduler.

### Verification performed

Run offline, against extracted source with stubs (no torch on the dev box):

* **`max_hash_depth_tokens=None` is byte-for-byte identical to the original**
  hasher across `hash_block_size` ∈ {16, 32, 64} and token counts
  {0, 15, 16, 17, 100, 999, 4000, 4001}. The off switch is provably a no-op.
* Cap halves the work: 4000 tokens at hash_block_size 16 goes from 250 hash
  computations to 125, and the capped list is an exact prefix of the uncapped one.
* Coverage is never short: `len(hashes) * hash_block_size >= cap` for every
  tested `(hash_block_size, cap)`, including non-multiples like 999 and 3000.
* Post-cap calls are O(1) no-ops — 50 repeated `update_block_hashes()` calls
  produced zero additional hash computations.
* **Alignment sweep** over `hash_block_size` × `block_size` (including
  `block_size > hash_block_size`, i.e. the `BlockHashListWithBlockSize` view) ×
  cap × prompt length × `num_cached_blocks`: no out-of-range index is ever
  reached, and the cap drops nothing the depth rule would have cached.

### Still not done

`pytest` has never run — there is no torch on this machine. The regression gate
in the Verification section above remains the first thing to execute on the GPU
box, in particular the `VLLM_SKIP_FILLER_HASHING=0` run, which must be green
unconditionally since that is the proof the whole change set is opt-in.
