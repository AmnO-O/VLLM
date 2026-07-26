# vLLM V1 Prefix Cache — Eviction Policy Research & Implementation Plan

**Target repo:** this checkout (`vllm` @ `e23b19309`, 2026-07-10, `main`)
**Target subsystem:** `vllm/v1/core/` (V1 engine)
**Workload:** AIC VChallenge multi-turn grading trace (`input_part2/trace_grading_public.jsonl`)
**Hardware:** 1× H200 (141 GB), 3 virtualized CPU cores

---

## 0. Corrections to the original brief

The source brief (`VChallenge/prefix_cache.md`) targets V0 files that **do not exist** in this
checkout:

| Brief says | Reality in this checkout |
|---|---|
| `vllm/core/block_manager_v1.py` | removed (V0) |
| `vllm/core/evictor_v1.py` | removed (V0) — **V1 has no `Evictor` class at all** |
| `vllm/core/policy.py` | removed (V0) |

Additionally, the brief's workload numbers disagree with the actual trace. The trace is
authoritative:

| Field | Brief / spec | Actual trace (420 rows) |
|---|---|---|
| Input tokens per turn | 1000 + 1000 + 150·turn (growing) | **~3999, flat at every turn** (turn 0..5 means: 4000, 3999, 3999, 3999, 3998, 3998) |
| Output tokens | 300 | **200** (`out_tokens_max`, uniform) |
| Conversations × turns | 70 × 6 | 70 × 6 ✓ |
| Arrival | Poisson seed 42 | spread over **302,866 ms** |
| Warmup | not mentioned | **90 / 420 rows** have `in_warmup: true` (conv 0–14, all 6 turns) → **330 graded requests** |
| Think time | not mentioned | 3000 ms, uniform |

Consequence: context does **not** grow across turns. Every prompt is ~4000 tokens. At
`--block-size=32` that is **125 blocks per request**.

---

## 1. Architecture map — where the policy actually lives

**The single most important structural fact:** V1 has no eviction policy *object*. The policy
**is** the ordering discipline of one intrusive doubly-linked free list. Victim selection is
`popleft()` from the head; "recency" is conferred by `append()` to the tail. Eviction is
**lazy** — a cached block's hash is dropped only at the moment the block is re-allocated.

### 1.1 File map

| Concern | Location |
|---|---|
| Block metadata | `vllm/v1/core/kv_cache_utils.py:117` — `KVCacheBlock` |
| **The policy** | `vllm/v1/core/kv_cache_utils.py:179` — `FreeKVCacheBlockQueue` |
| Eviction trigger | `vllm/v1/core/block_pool.py:625` — `BlockPool.get_new_blocks` |
| Eviction execution | `vllm/v1/core/block_pool.py:657` — `BlockPool._maybe_evict_cached_block` |
| Hit → pin | `vllm/v1/core/block_pool.py:680` — `BlockPool.touch` |
| Free → re-insert | `vllm/v1/core/block_pool.py:697` — `BlockPool.free_blocks` |
| Hit lookup | `vllm/v1/core/block_pool.py:198` — `BlockPool.get_cached_block` |
| Hash map | `vllm/v1/core/block_pool.py:33` — `BlockHashToBlockMap` |
| Cache insert | `vllm/v1/core/block_pool.py:225` — `BlockPool.cache_full_blocks` |
| Hit-length search | `vllm/v1/core/single_type_kv_cache_manager.py` — `find_longest_cache_hit` |
| Per-request orchestration | `vllm/v1/core/kv_cache_manager.py:207` / `:269` / `:490` |
| Free-order reversal | `vllm/v1/core/sched/scheduler.py:2163` |

### 1.2 `KVCacheBlock` — available metadata

```python
@dataclass(slots=True)                     # kv_cache_utils.py:117
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0                       # >0 ⇒ pinned, not in free list
    _block_hash: BlockHashWithGroupId | None = None
    _block_hash_num_tokens: int | None = None   # ← prefix DEPTH in tokens
    prev_free_block: "KVCacheBlock | None" = None   # intrusive list links
    next_free_block: "KVCacheBlock | None" = None
    is_null: bool = False
```

Note what is **absent**: no timestamp, no access counter, no frequency estimate. Adding any of
these costs memory per block *and* CPU per touch. `slots=True` means adding a field is cheap in
bytes but the class is on the hottest CPU path in the scheduler.

Note what is **present and underused**: `_block_hash_num_tokens` (accessor at
`kv_cache_utils.py:145`) already encodes how deep into the prefix a block sits. This is free
signal for a depth-aware policy — no new metadata required.

### 1.3 Lifecycle call graph

```
Request arrives
   │
   ▼
Scheduler.schedule()                                    sched/scheduler.py
   │
   │  ── HASHING (done eagerly at request construction) ──
   │     get_request_block_hasher()                     kv_cache_utils.py:687
   │       └─ hash_block_tokens()                       kv_cache_utils.py:591
   │            xxhash (default) | sha256   → request.block_hashes
   │
   ├─► KVCacheManager.get_computed_blocks(request)      kv_cache_manager.py:207
   │      ▼
   │   KVCacheCoordinator.find_longest_cache_hit()      kv_cache_coordinator.py
   │      ▼
   │   BlockPool.get_cached_block(hash, group_ids)      block_pool.py:198
   │      ▼
   │   BlockHashToBlockMap.get_one_block()              block_pool.py:61
   │      │
   │      ├─ HIT ──► BlockPool.touch(blocks)            block_pool.py:680
   │      │            ├─ FreeKVCacheBlockQueue.remove(block)   O(1) unlink
   │      │            └─ block.ref_cnt += 1            ← PINNED, now unevictable
   │      │
   │      └─ MISS ─► fall through to allocation
   │
   ├─► KVCacheManager.allocate_slots(...)               kv_cache_manager.py:269
   │      ▼
   │   BlockPool.get_new_blocks(n)                      block_pool.py:625
   │      ├─ FreeKVCacheBlockQueue.popleft_n(n)         kv_cache_utils.py:268
   │      │     ══════ THE EVICTION DECISION ══════
   │      │     head of list == LRU victim. No scan, no comparison, no heap.
   │      └─ for b in ret: _maybe_evict_cached_block(b) block_pool.py:657
   │            └─ _remove_cached_block_hashes(b)       block_pool.py:567
   │                 └─ drops b's hash from the map     ← cache entry dies HERE
   │
   ├─► BlockPool.cache_full_blocks(...)                 block_pool.py:225
   │      └─ _insert_block_hash()                       block_pool.py:603
   │           inserts every newly-full block into the hash map
   │
   ├─► (decode ×200 steps; PagedAttention reads block_ids)
   │
   └─► Request finishes
          ▼
       Scheduler._free_request_blocks()                 sched/scheduler.py:2136
          │
          ├─ sync path (defer_block_free == False):
          │     KVCacheManager.free() ─► coordinator ─► single_type_kv_cache_manager.py:411
          │        block_pool.free_blocks(reversed(pop_blocks_for_free(...)))
          │
          └─ async path (--async-scheduling ⇒ defer_block_free == True):
                pop_blocks_for_free(request) ─► self.deferred_frees
                   └─ _drain_deferred_frees()           sched/scheduler.py:2151
                        block_pool.free_blocks(reversed(blocks))    :2163
                          ▼
                BlockPool.free_blocks(ordered_blocks)   block_pool.py:697
                   ├─ ref_cnt -= 1
                   ├─ if hash is None → prepend_n()  ← HEAD: die first
                   └─ else            → append_n()   ← TAIL: survive longest
```

**The current submission enables `--async-scheduling`, so the deferred path at
`scheduler.py:2163` is the live one.** Both paths converge on `free_blocks`, so a policy change
in `free_blocks` / `FreeKVCacheBlockQueue` covers both.

### 1.4 The current policy, stated precisely

> **LRU over free-insertion order, with two refinements:**
> 1. **Unhashed-first** (`block_pool.py:717`): blocks that can never produce a cache hit are
>    `prepend_n`'d to the head, so they are consumed before any cached block is sacrificed.
> 2. **Reverse-depth within a request** (`scheduler.py:2163`): `reversed(blocks)` means the
>    deepest (tail) blocks are appended first and therefore sit **nearer the head** than the
>    request's shallow prefix blocks. Prefix blocks are appended last and sit nearest the tail.

Complexity: `popleft` / `remove` / `append` are all **O(1)** and allocate no Python objects
(`kv_cache_utils.py:184-186` documents this as deliberate). This is the CPU budget any
replacement policy must respect on 3 cores.

---

## 2. Instrumentation plan

Hook points, in order of value. All are cheap counters; guard them behind a flag so the
measurement build and the submission build can differ.

| Metric | Hook site | Note |
|---|---|---|
| **Evictions of live cache entries** | `block_pool.py:657` `_maybe_evict_cached_block`, count only when it returns `True` | **This is the gate metric.** If it stays 0, no policy change can help. |
| Eviction depth histogram | same site, bucket on `block.block_hash_num_tokens` | distinguishes "shared prefix evicted" (depth < 1000) from "filler evicted" |
| Cache hits / reuse depth | `block_pool.py:680` `touch` | length of `blocks` = blocks reused by this request |
| Hit rate (built-in) | `kv_cache_manager.py:195` `make_prefix_cache_stats` | already exists; surfaces as `gpu_prefix_cache_hit_rate` |
| Allocation pressure | `block_pool.py:625` `get_new_blocks` | log `get_num_free_blocks()` before pop → free-block low-water mark |
| Hash-map churn | `block_pool.py:603` `_insert_block_hash` vs `:567` `_remove_cached_block_hashes` | insert/remove ratio ⇒ how many cached blocks never serve a hit |
| Scheduler CPU | wall-clock around `Scheduler.schedule()` | the 3-core bottleneck; also `--kv-cache-metrics` exists as a flag |
| TTFT / TPOT p50,p95 | client side, from the replay harness | exclude the 90 `in_warmup` rows |

Existing infrastructure to reuse rather than reinvent: `vllm/v1/core/kv_cache_metrics.py`
(`KVCacheMetricsCollector`, already wired into `BlockPool` via the `metrics_collector` ctor arg
at `block_pool.py:168`, with call sites at `:647`, `:670`, `:694`).

---

## 3. Trace bottleneck analysis

### 3.1 Block-level structure of one request

At `--block-size=32`, a ~4000-token prompt = **125 blocks**:

| Segment | Tokens | Blocks | Reuse property |
|---|---|---|---|
| Shared system prefix | 1000 | ~31 | shared by **all 70** conversations |
| Per-conversation prefix | 1000 | ~31 | shared by the **6 turns** of one conversation |
| Turn-local remainder | ~2000 | ~63 | **unique to this turn — will never be hit again** |

So ~50% of every prefill is theoretically reusable, and **~63 of 125 blocks per request are
write-once-read-never**.

### 3.2 Why LRU already handles the shared prefix well

The shared system prefix is touched by every arriving request. `touch()`
(`block_pool.py:680`) removes those blocks from the free list and sets `ref_cnt > 0`. With 70
conversations arriving across 303 s and only 3000 ms of think time between turns, **at least one
request is almost always referencing the shared prefix**, so those ~31 blocks are
near-permanently pinned and are not eviction candidates at all. The brief's worry — "do 1-hit
turn contexts evict the 1000-token system prefix?" — is largely answered by reference counting,
not by the eviction policy.

### 3.3 Why LRU already handles filler-vs-conversation-prefix well

Within one freed request, `reversed(blocks)` (`scheduler.py:2163`) appends the 63 turn-local
filler blocks *before* the 31 conversation-prefix blocks. Filler therefore lands closer to the
head and is evicted first. **The reverse-depth heuristic is already doing the right thing for
this exact workload.** This substantially reduces the expected upside of swapping in
SLRU/LFU/ARC.

### 3.4 The genuine remaining inefficiencies

Ranked by expected value on this trace:

1. **Write-once-read-never blocks are still hashed and inserted.** ~63 blocks per request ×
   330 graded requests ≈ **20,800 pointless hash-map inserts**, each later requiring a removal
   at eviction (`block_pool.py:567`). On 3 cores this is pure GIL-bound scheduler overhead that
   buys nothing. This is a **CPU** win, available regardless of whether eviction ever fires.
2. **Conversation prefixes are unpinned during the 3000 ms think gap.** For conversation *C*,
   between turn *t* finishing and turn *t+1* arriving, nothing references *C*'s prefix, so its
   ~31 blocks sit in the free list as eviction candidates. This is the one population a
   depth-aware policy could protect. It is also the population whose loss costs the most: a miss
   forces re-prefill of 1000 tokens.
3. **Cross-turn recency inversion.** LRU has no notion that a conversation prefix has a
   *scheduled* future reuse 3000 ms out, whereas filler has none. Lifecycle/turn-awareness is
   the theoretically correct lens here — but see §3.5.

### 3.5 The gate: does eviction happen at all?

This determines whether §3.4 items 2–3 are worth any code.

- Working set: 70 conversations × ~4200 tokens ≈ **294 K tokens**, hard-bounded by
  `--max-model-len=4800`.
- Available KV memory: `--gpu-memory-utilization=0.95` × 141 GB ≈ 134 GB, less ~1.2 GB for a
  1.2 B model at fp8 ⇒ **~132 GB** for KV at `--kv-cache-dtype=fp8`.
- LFM2.5-1.2B is a Liquid hybrid: predominantly convolution/SSM layers with only a few
  full-attention layers. KV bytes per token is therefore **much smaller** than a comparable
  dense transformer, and the conv/SSM state is handled by a separate non-prefix-cacheable
  spec (`MambaSpec` path in `single_type_kv_cache_manager.py`).

**Expectation: free blocks vastly exceed the working set and the eviction branch never fires.**
This is an unverified prediction, not a measurement. It must be confirmed in Phase 0 below
before any policy code is written, because if it holds, items 2–3 are worth exactly zero and
item 1 (CPU) is the entire prize.

---

## 4. Candidate policy matrix

Ratings are **analytical**, derived from the trace structure in §3 and the O(1) constraint of
`FreeKVCacheBlockQueue` — they are not benchmark results.

| Policy | Expected hit-rate gain vs baseline | Victim-select cost | Per-touch cost | Extra bytes/block | Fits V1 intrusive list? | Verdict |
|---|---|---|---|---|---|---|
| **LRU + reverse-depth (baseline)** | — | O(1) | O(1) | 0 | native | incumbent; already prefix-biased (§3.3) |
| **Segmented LRU (SLRU)** | small–moderate (protects §3.4 item 2) | O(1) | O(1) | 0 (reuse `_block_hash_num_tokens`) | **yes** — two lists, same primitives | **best fit** |
| LFU | low | O(1) w/ bucket lists | O(1) + counter write | 4–8 | awkward: needs bucket array | poor value |
| LRU-2 | low | O(1) | 2 timestamp writes | 16 | needs history map for evicted keys | not worth it |
| W-TinyLFU | low here (workload has no scan/one-hit-wonder storm beyond filler, already handled) | O(1) | sketch update ×4 hashes | sketch is global (~KB) | doable but heavy | CPU cost > benefit on 3 cores |
| ARC | low | O(1) | ghost-list bookkeeping | 0 + 2 ghost lists | 4 lists to maintain | complexity unjustified |
| LIRS | low | O(1) amortised | reuse-distance tracking | 8–16 | stack pruning is not O(1) worst case | reject on CPU grounds |
| Token-weighted LRU/LFU | moderate *in principle* | O(log N) if by weight | O(1) | 4 | **breaks O(1)** — needs heap | reject: O(log N) on 3 cores |
| **Turn-aware / lifecycle** | highest *ceiling* | O(1) | O(1) | 4 (conv id / turn idx) | yes, but needs plumbing from `Request` | **highest upside, highest risk** |
| **Don't-cache-unreusable** (not an eviction policy) | 0 hit-rate, **real CPU win** | n/a | removes work | 0 | yes | **do this regardless** (§3.4 item 1) |

The reject column is doing most of the work here. Any policy requiring a heap, a sorted
structure, or a per-touch sketch update is disqualified by the 3-core CPU budget *before* its
hit-rate merits are considered — the whole point of `FreeKVCacheBlockQueue` is that it is
allocation-free and O(1).

---

## 5. Top-3 deep dive

### 5.1 Segmented LRU (protected / probationary) — **recommended**

**Formulation.** Partition the free list into two lists, *probationary* `P` and *protected* `Q`.
On free, classify block *b*:

```
depth(b) = b.block_hash_num_tokens          # already stored, no new metadata
segment(b) = Q  if b.block_hash is not None and depth(b) <= SHARED_PREFIX_TOKENS
             P  otherwise
```

Victim selection: drain `P` from the head; only when `P` is empty drain `Q` from the head.
Optionally cap `|Q| <= alpha * num_gpu_blocks` (alpha ≈ 0.25) and demote `Q`'s head to `P`'s
tail on overflow, so a pathological workload cannot starve `P`.

**Metadata overhead:** **0 bytes per block.** `_block_hash_num_tokens` already exists; segment
membership is implied by which list the block is linked into. This is the decisive advantage.

**Cost:** all operations remain O(1). `remove()` must know which list to decrement, which needs
either a 1-bit flag on the block or two `num_free_blocks` counters plus a lookup — the flag is
simpler and fits in the existing `slots` layout.

**Modifications:** `FreeKVCacheBlockQueue` gains a second head/tail pair and segment-aware
`popleft_n` / `remove`; `BlockPool.free_blocks` (`block_pool.py:697`) gains the classification
branch. `get_num_free_blocks()` (`block_pool.py:775`) must return the **sum**.

**Failure mode:** if `SHARED_PREFIX_TOKENS` is mistuned, `Q` fills with junk and the policy
degrades to LRU with extra branches — i.e. it fails *safe*, which is why it is the
recommendation. Second failure mode: `get_num_free_blocks()` desync ⇒ `popleft` raises
`ValueError("No free blocks available")` ⇒ **failed requests**, which violates the
`failed_count == 0` success criterion. Test this specifically.

### 5.2 Turn-aware / conversation-lifecycle eviction

**Formulation.** Score by *expected time to next reuse* rather than time since last use. A
conversation prefix freed at time *t* has a known reuse at ≈ *t* + `think_ms`; filler has none.
Protect blocks whose predicted reuse is imminent:

```
priority(b) = 0                      if depth(b) > per_conv_prefix_end   (filler)
              1                      if conversation is mid-dialogue
              2                      if depth(b) <= shared_prefix_end
```

Three lists, drained 0 → 1 → 2. Still O(1).

**Metadata:** needs conversation identity, which the KV cache layer **does not have**.
`BlockPool` sees `Request` objects (`cache_full_blocks` takes one) but conversation grouping is
an application-level concept. You would have to infer it from the hash chain (blocks sharing the
conversation-prefix hash belong to the same conversation) — which is elegant, since the hash map
already groups them, or plumb a field through `Request`. Est. 4 bytes/block if plumbed.

**Pros:** highest theoretical ceiling; directly targets §3.4 item 2.
**Cons:** it encodes trace-specific assumptions (uniform 3000 ms think time, 6 turns). It is
tuned to the public trace and may not transfer to the private grading trace. **Overfitting risk
is the main argument against.**

### 5.3 Don't-cache-provably-unreusable blocks

Not an eviction policy, but the highest expected value per LOC on this workload, and the only
candidate whose benefit is **independent of the Phase 0 gate**.

**Formulation.** At `cache_full_blocks` (`block_pool.py:225`), skip `_insert_block_hash` for
blocks that cannot serve a future hit. The mechanism already exists: the `block_mask` parameter
(`block_pool.py:233`, documented at `:251-257`) was added for exactly this purpose (SWA
tail-window groups) — "blocks that can never serve a hit stay out of the prefix-cache hash map."

**Benefit:** removes ~20,800 map inserts and their matching removals from the scheduler's
critical section (§3.4 item 1). Pure CPU reduction on the 3-core bottleneck, zero hit-rate cost
*if* the "unreusable" predicate is sound.

**Risk:** the predicate is only sound if turn-local content truly never repeats. If the private
trace has repeated suffixes, this **loses** hits. Mitigation: gate on depth
(`> shared + per_conv prefix`) rather than on any content assumption, and make the threshold a
config knob so it can be disabled without a rebuild.

### 5.4 Final recommendation

**Ship §5.3 unconditionally; ship §5.1 only if Phase 0 shows non-zero evictions; treat §5.2 as
research only.** §5.3 pays off on the measured bottleneck (3 cores, GIL-bound scheduler). §5.1
is a fail-safe, zero-metadata, O(1)-preserving change. §5.2 overfits the public trace.

---

## 6. Implementation roadmap

Ordered. Each step is independently revertable.

| # | Step | Files | Est. LOC | Risk |
|---|---|---|---|---|
| 0 | **Gate.** Compute KV bytes/token for LFM2.5-1.2B from its `config.json`; instrument `_maybe_evict_cached_block` and replay the trace. **If evictions ≈ 0, skip steps 3–5.** | `block_pool.py:657` | ~20 (temporary) | none |
| 1 | Wire the metrics from §2 behind a flag | `kv_cache_metrics.py`, `block_pool.py` | ~60 | low |
| 2 | **§5.3** depth-gated cache-insert skip, config-knobbed | `block_pool.py:225`, `:603` | ~40 | low — reuses existing `block_mask` |
| 3 | Offline replay simulator over the extracted block-hash reference stream | new `VChallenge/sim_evict.py` | ~250 | none (offline) |
| 4 | **§5.1** segmented `FreeKVCacheBlockQueue` | `kv_cache_utils.py:179` | ~80 | **medium — see below** |
| 5 | Segment classification on free | `block_pool.py:697`, `:775` | ~30 | medium |
| 6 | Unit tests: free-list invariants, counter sync, `reset_prefix_cache` | `tests/v1/core/` | ~150 | — |

**Data structures:** no new ones. Step 4 is a second head/tail pair over the *existing*
intrusive links, plus a 1-bit segment flag on `KVCacheBlock`. No heaps, no sketches, no
timestamps — this is the whole point.

**Concurrency & thread-safety:** the V1 scheduler and `BlockPool` run **single-threaded inside
the engine core process**; `FreeKVCacheBlockQueue` takes no locks and needs none. Do **not** add
any. The relevant hazard is not a data race but the **async-scheduling deferred-free window**
(`scheduler.py:2140-2149`): when `defer_block_free` is true, blocks are held in
`self.deferred_frees` and are *not* in the free list, so `get_num_free_blocks()` legitimately
under-reports until `_drain_deferred_frees()` runs. Any invariant assertion you add must tolerate
that window, or you will trip asserts under `--async-scheduling` — which the submission enables.

**Correctness invariants to assert in tests:**
- `sum(num_free_blocks per segment) == get_num_free_blocks()`
- every block with `ref_cnt == 0 and not is_null` is linked into exactly one segment
- every block with `ref_cnt > 0` is linked into none
- `reset_prefix_cache()` (`block_pool.py:739`) still returns True when only the null block is held

---

## 7. Benchmark & validation methodology

**Replay harness.** The grading client is organizer-side and is not in this checkout; only the
trace and the spec are. A local replay harness must be written. Parameters to honour from the
trace: 70 conversations × 6 turns, ~4000-token prompts, `max_tokens=200`, 3000 ms think time
between a conversation's turns, per-row `timestamp_ms` arrival (max 302,866 ms), OpenAI
chat-completions endpoint on `:8000`.

**Scoring hygiene — do not skip this:** the 90 rows with `in_warmup: true` (conversations 0–14)
must be **excluded from all latency statistics**. Including them will make every policy look
identical, because warmup is dominated by cold-start prefill.

**Ablations.** Run each independently, not stacked:
1. baseline (current `docker-compose-16.yml`)
2. baseline + step 2 (cache-insert skip) — isolates the CPU win
3. baseline + steps 4–5 (SLRU) — isolates the hit-rate win
4. both
5. `--enable-prefix-caching` **off** — establishes how much prefix caching is worth here at all,
   and bounds every number above

Ablation 5 is the cheapest sanity check and should be run first; if it barely moves TTFT, the
entire prefix-cache axis is not where the score is.

**Separate the two hit populations** (per the brief's Phase 7): report shared-prefix hits and
conversation-prefix hits as distinct series, using the eviction-depth histogram from §2. An
aggregate hit rate will hide the only effect the policy change can have.

**Success criteria.** In priority order:
1. `failed_count == 0` — a hard gate. A free-list accounting bug turns into
   `ValueError("No free blocks available")` (`kv_cache_utils.py:245`) and fails requests, which
   costs more than any latency gain earns.
2. TTFT p50 < 50 ms on graded (non-warmup) requests.
3. TTFT p95 reduction — the stated pain point.
4. ERS maximisation.

**Repeat count:** ≥3 runs per configuration. With Poisson arrivals over 303 s on 3 CPU cores,
run-to-run variance will likely exceed the effect size of a policy change; a single run cannot
distinguish them.

---

## 8. Open questions

1. **Does eviction fire?** (§3.5) Gates steps 3–5. Highest priority.
2. **Deployment path.** The submission is a `docker-compose.yml` against the published
   `vllm/vllm-openai:v0.22.1`. Source edits in this checkout reach the container only via a
   volume mount over `site-packages` or a custom image build. This checkout is `main` @
   2026-07-10, not `v0.22.1`; `v1/core` APIs differ (`BlockHashToBlockMap`, `prepend_n`,
   `iter_blocks_after`, `metrics_collector` are all recent), so mounting these files into a
   `v0.22.1` image will fail at import. Per direction, work proceeds in this checkout — but the
   mount-vs-build decision must be settled before any of this is measurable end-to-end.
3. **Hybrid-model interaction.** LFM2.5 mixes attention and conv/SSM layers. Confirm how many
   KV cache groups exist and whether the conv-state group participates in prefix caching at all
   — if the attention group is a small minority of layers, the memory pressure premise weakens
   further.
4. **Private-trace transferability.** §5.2 and the depth thresholds in §5.1/§5.3 are tuned to the
   public trace's 1000/1000/2000 token layout. Anything keyed to those constants must be a config
   knob, not a hardcoded literal.
