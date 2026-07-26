====================================================
vLLM Prefix Cache Optimization: Master Research & Implementation Plan
====================================================

[ROLE & PERSONA]
You are a Principal Systems Engineer & LLM Infrastructure Researcher specializing in GPU memory optimization, vLLM internals, PagedAttention, and eviction algorithm design.

[TARGET WORKLOAD TRACE]
{
  "workload": "vLLM multi-turn (shared prefix + growing context, output length pinned)",
  "num_conversations": 70,
  "user_turns_per_conversation": 6,
  "total_requests": 420,
  "shared_system_prefix_tokens": 1000,
  "per_conversation_prefix_tokens": 1000,
  "new_user_tokens_per_turn": 150,
  "output_tokens_per_turn_pinned": 300,
  "arrival": "Poisson, seed 42"
}

[HARDWARE & SYSTEM CONSTRAINTS]
- CPU: 3 Core Virtualized System (High CPU GIL / Scheduling bottleneck)
- GPU: NVIDIA H200 (141GB VRAM, High compute capacity)
- Core Issue: High tail latency (TTFT P95) caused by CPU scheduling overhead and sub-optimal KV Cache eviction under high concurrent load.

====================================================
RESEARCH & IMPLEMENTATION PHASES
====================================================

Follow this strict step-by-step workflow:

----------------------------------------------------
Phase 1 — Reverse Engineering & Architecture Mapping
----------------------------------------------------
- Map the exact vLLM source code files responsible for Prefix Caching and Eviction (e.g., `vllm/core/block_manager_v1.py`, `vllm/core/evictor_v1.py`, `vllm/core/policy.py`).
- Trace the complete lifecycle of a prefix block: Allocation -> Hashing (xxHash/SHA256) -> Hit/Miss Lookup -> Pinning/Reference Counting -> Eviction.
- Explain interactions between BlockManager, BlockAllocator, KV Cache, Scheduler, and PagedAttention.
- Output: A clear ASCII/Text Call Graph & Sequence Workflow.

----------------------------------------------------
Phase 2 — Instrumentation & Metric Collection Design
----------------------------------------------------
Specify the EXACT Python files and methods in vLLM codebase where each of the following metrics must be hooked:
- Cache Hit/Miss Rate, Prefix Reuse Count, Saved Prefill Tokens, Saved FLOPs.
- CPU Scheduling Overhead (GIL time spent in Evictor), Lock Contention, Block Fragmentation.
- Latency Metrics: TTFT (p50/p95), TPOT (p50/p95), ERS (Effective Request Score).

----------------------------------------------------
Phase 3 — Trace Bottleneck Analysis
----------------------------------------------------
Analyze the baseline LRU policy against the provided 420-request JSON trace:
- Identify cache pollution risks (e.g., do 1-hit turn contexts evict 1000-token system/conversation prefixes?).
- Evaluate Temporal Locality vs. Frequency Locality vs. Lifecycle Locality in this trace.
- Quantify TTFT lost due to recomputing shared prefixes vs per-conversation prefixes.

----------------------------------------------------
Phase 4 — Candidate Eviction Policies Evaluation
----------------------------------------------------
Evaluate candidate algorithms under this specific workload trace:
1. LRU (Baseline)
2. Segmented LRU (SLRU)
3. LFU
4. LRU-2
3. W-TinyLFU
4. ARC (Adaptive Replacement Cache)
5. LIRS
6. Token-Weighted LRU/LFU
7. Conversation Lifecycle-Aware Cache (Turn-Aware Eviction)
8. Hybrid Frequency-Recency Pinning Cache

Provide a structured Markdown Matrix comparing all candidates across:
- Expected Cache Hit Rate | TTFT Impact | CPU Overhead (O(1) vs O(log N)) | vLLM Compatibility

----------------------------------------------------
Phase 5 — Algorithm Selection & Deep Dive
----------------------------------------------------
Select the TOP 3 candidate policies for implementation.
For each candidate, detail:
- Mathematical/Algorithmic Formulation
- Metadata overhead (Bytes per KV Block)
- Required modifications in `vllm/core/evictor_v1.py` and `block_manager`
- Pros, Cons, and Failure Modes under 3 Core CPU constraint.
- Final Recommendation: Single best algorithm for this trace.

----------------------------------------------------
Phase 6 — Implementation & Modification Roadmap
----------------------------------------------------
Provide a step-by-step implementation plan:
- File paths to modify
- Estimated Lines of Code (LOC)
- Python data structures needed (e.g., Doubly Linked Lists, Radix Trees, Hash Maps)
- Concurrency & Thread-Safety considerations for Async Engine.

----------------------------------------------------
Phase 7 — Benchmark & Validation Methodology
----------------------------------------------------
Design the benchmark protocol:
- Trace replay harness script parameters.
- Ablation study plan (Testing Shared Prefix Hit vs Conv Prefix Hit).
- Success Criteria: Zero failed requests (`failed_count = 0`), TTFT p50 < 50ms, ERS score maximization.

====================================================
OUTPUT INSTRUCTIONS
====================================================
- Produce a dense, rigorous, engineering-grade response.
- Use clear Markdown headings, code blocks, and comparison tables.
- Do NOT skip technical details. Focus heavily on CPU efficiency given the 3 Core limit.

[FILE OUTPUT ACTION]
Save the complete output directly into a new markdown file named:
`docs/PREFIX_CACHE_RESEARCH_PLAN.md`

Ensure the file is self-contained, fully formatted, and ready to be committed to Git.

