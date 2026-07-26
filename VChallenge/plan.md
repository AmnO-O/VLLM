Please execute the implementation plan for the V1 Prefix Cache optimization. 

**STRICT REQUIREMENT:** Every single feature, threshold, and strategy MUST be fully configurable via CLI flags (passed to `vllm serve`) and/or Environment Variables so they can be easily tuned or toggled directly inside `docker-compose.yml` without modifying source code.

---

### 1. Configuration & Flag Definitions
Add the following configuration options into engine args (e.g., `vllm/engine/arg_utils.py` or fallback to `os.getenv`):

- `--enable-skip-filler-hashing` / `VLLM_SKIP_FILLER_HASHING` (bool, default: `True`): Toggles skipping hash insertion for turn-local filler blocks.
- `--filler-depth-tokens` / `VLLM_FILLER_DEPTH_TOKENS` (int, default: `2000`): Token depth threshold beyond which blocks are treated as write-once filler.
- `--prefix-cache-policy` / `VLLM_PREFIX_CACHE_POLICY` (str, options: `lru`, `slru`, `flu`, default: `lru`): Selects the free-list eviction ordering policy.
- `--slru-protected-tokens` / `VLLM_SLRU_PROTECTED_TOKENS` (int, default: `1000`): Token depth boundary for the protected segment under SLRU.

---

### 2. Phase 0: Gate Check & Eviction Instrumentation
1. Add temporary counter/logging inside `vllm/v1/core/block_pool.py:_maybe_evict_cached_block` to track whether active cache eviction ever occurs.
2. Build a standalone Python script `replay_trace.py` that parses `input_part2/trace_grading_public.jsonl` and replays all 420 requests adhering to exact `timestamp_ms` arrival times and 3000 ms think gaps across turns.
3. Run the replay against the H200 server to measure and report the total number of evictions (`eviction_count`).

---

### 3. Implementation: Solution 5.3 (Skip Filler Hashing / CPU Optimization)
1. In `vllm/v1/core/block_pool.py` (`cache_full_blocks` & `_insert_block_hash`), check if `--enable-skip-filler-hashing` is enabled.
2. If `block_hash_num_tokens > filler_depth_tokens`, skip `_insert_block_hash` for that block.
3. Ensure this eliminates pointless hash-map inserts/removals for turn-local filler while preserving shared/conversation prefixes.

---

### 4. Implementation: Solution 5.1 (Segmented LRU - SLRU)
1. In `vllm/v1/core/kv_cache_utils.py` (`FreeKVCacheBlockQueue`), implement two internal segment lists (Protected and Probationary) when `--prefix-cache-policy=slru`.
2. In `block_pool.py:free_blocks`, classify blocks into Protected (`depth <= slru_protected_tokens`) or Probationary (`depth > slru_protected_tokens`).
3. Maintain $O(1)$ operations with zero extra bytes allocated per block.

---

### 5. Validation & Metrics
1. Re-run `replay_trace.py` and benchmark Baseline vs. Optimization configurations.
2. **CRITICAL:** Exclude the 90 warmup rows (`in_warmup: true`, conversations 0–14) from all latency statistics.
3. Output the TTFT (P50, P95) and TPOT metrics. Confirm `failed_count == 0` (no dropped requests or accounting errors).
4. Provide a complete `docker-compose.yml` snippet demonstrating how to configure these flags/env vars.