# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay the VChallenge multi-turn grading trace against a vLLM server.

Trace semantics (verified against ``input_part2/trace_grading_public.jsonl``):

* 420 rows = 70 conversations x 6 turns.
* ``timestamp_ms`` is only meaningful on ``turn_idx == 0`` — it is the
  conversation's Poisson arrival time (spread over ~303 s). Later turns carry
  ``timestamp_ms == 0`` and are driven by ``think_ms`` after the previous
  response completes.
* ``in_warmup`` is constant per conversation and true for conversations 0-14
  (90 rows). Those rows are **excluded from latency statistics** — including
  them washes out any policy difference, because warmup is dominated by
  cold-start prefill.
* Every prompt is ~4000 tokens (~12000 chars) and every response is pinned to
  ``out_tokens_max`` (200) tokens. Context does *not* grow across turns.

Prompt construction mirrors the workload spec so that prefix caching is actually
exercised:

    [ shared system prefix ][ per-conversation prefix ][ turn-local filler ]
      identical for all          identical across          unique per turn,
      420 requests               a conversation's turns    never reused

Usage:

    pip install aiohttp
    python VChallenge/replay_trace.py --base-url http://localhost:8000 \
        --model LFM2.5-1.2B-Instruct --trace ../input_part2/trace_grading_public.jsonl

Add ``--speedup 10`` for a quick smoke run (compresses the 303 s arrival
schedule; latency numbers are then not comparable to a real run).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

try:
    import aiohttp
except ImportError:  # pragma: no cover
    sys.exit("replay_trace.py needs aiohttp. Install it with: pip install aiohttp")

# Chars-per-token observed in the trace (12000 chars / 4000 tokens).
CHARS_PER_TOKEN = 3.0

# Deterministic filler vocabulary. Words are 3 chars + 1 space = 4 chars, so the
# generated text length is predictable and stable across runs.
_WORDS = [f"{i:03d}" for i in range(1000)]


def _make_text(seed: int, num_chars: int) -> str:
    """Build deterministic text of approximately `num_chars` characters.

    The same seed always yields the same text, which is what makes the shared
    and per-conversation prefixes cacheable.
    """
    if num_chars <= 0:
        return ""
    out: list[str] = []
    total = 0
    i = seed
    while total < num_chars:
        w = _WORDS[i % len(_WORDS)]
        out.append(w)
        total += len(w) + 1
        # Cheap deterministic walk; avoids importing random for reproducibility.
        i = (i * 1103515245 + 12345) & 0x7FFFFFFF
    return " ".join(out)[:num_chars]


@dataclass
class TurnSpec:
    conv_id: int
    turn_idx: int
    in_warmup: bool
    timestamp_ms: int
    think_ms: int
    in_chars: int
    out_tokens_max: int


@dataclass
class TurnResult:
    conv_id: int
    turn_idx: int
    in_warmup: bool
    ok: bool
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    total_ms: float | None = None
    output_tokens: int = 0
    error: str | None = None


@dataclass
class PromptLayout:
    """Byte budget for each prompt segment, derived from the workload spec."""

    shared_prefix_tokens: int = 1000
    conv_prefix_tokens: int = 1000

    @property
    def shared_chars(self) -> int:
        return int(self.shared_prefix_tokens * CHARS_PER_TOKEN)

    @property
    def conv_chars(self) -> int:
        return int(self.conv_prefix_tokens * CHARS_PER_TOKEN)

    def build(self, spec: TurnSpec) -> str:
        shared = _make_text(seed=1, num_chars=self.shared_chars)
        conv = _make_text(seed=1_000_000 + spec.conv_id, num_chars=self.conv_chars)
        filler_chars = max(0, spec.in_chars - self.shared_chars - self.conv_chars)
        filler = _make_text(
            seed=5_000_000 + spec.conv_id * 97 + spec.turn_idx, num_chars=filler_chars
        )
        return f"{shared}\n{conv}\n{filler}"


def load_trace(path: str) -> dict[int, list[TurnSpec]]:
    convs: dict[int, list[TurnSpec]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            convs[r["conv_id"]].append(
                TurnSpec(
                    conv_id=r["conv_id"],
                    turn_idx=r["turn_idx"],
                    in_warmup=bool(r["in_warmup"]),
                    timestamp_ms=int(r["timestamp_ms"]),
                    think_ms=int(r["think_ms"]),
                    in_chars=int(r["in_chars"]),
                    out_tokens_max=int(r["out_tokens_max"]),
                )
            )
    for turns in convs.values():
        turns.sort(key=lambda t: t.turn_idx)
    return dict(convs)


async def run_turn(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    layout: PromptLayout,
    spec: TurnSpec,
) -> TurnResult:
    payload = {
        "model": args.model,
        "prompt": layout.build(spec),
        "max_tokens": spec.out_tokens_max,
        "temperature": 0.0,
        "stream": True,
        # Pin the output length exactly, as the workload spec requires.
        "ignore_eos": True,
        "stream_options": {"include_usage": False},
    }
    start = time.perf_counter()
    first_token_at: float | None = None
    n_tokens = 0
    try:
        async with session.post(
            f"{args.base_url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=args.request_timeout),
        ) as resp:
            if resp.status != 200:
                body = (await resp.text())[:200]
                return TurnResult(
                    spec.conv_id,
                    spec.turn_idx,
                    spec.in_warmup,
                    ok=False,
                    error=f"HTTP {resp.status}: {body}",
                )
            async for raw in resp.content:
                if not raw.startswith(b"data:"):
                    continue
                data = raw[len(b"data:") :].strip()
                if data == b"[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices or not choices[0].get("text"):
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                n_tokens += 1
    except Exception as exc:  # noqa: BLE001 - report, never abort the run
        return TurnResult(
            spec.conv_id,
            spec.turn_idx,
            spec.in_warmup,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    end = time.perf_counter()
    if first_token_at is None:
        return TurnResult(
            spec.conv_id,
            spec.turn_idx,
            spec.in_warmup,
            ok=False,
            error="no tokens received",
        )
    ttft_ms = (first_token_at - start) * 1000
    decode_ms = (end - first_token_at) * 1000
    tpot_ms = decode_ms / max(1, n_tokens - 1) if n_tokens > 1 else None
    return TurnResult(
        spec.conv_id,
        spec.turn_idx,
        spec.in_warmup,
        ok=True,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_ms=(end - start) * 1000,
        output_tokens=n_tokens,
    )


async def run_conversation(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    layout: PromptLayout,
    turns: list[TurnSpec],
    t_zero: float,
    results: list[TurnResult],
    sem: asyncio.Semaphore | None,
) -> None:
    # Turn 0 carries the conversation's arrival time; later turns are driven by
    # think_ms after the previous response completes.
    arrival_s = turns[0].timestamp_ms / 1000.0 / args.speedup
    delay = t_zero + arrival_s - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)

    for i, spec in enumerate(turns):
        if i > 0:
            await asyncio.sleep(spec.think_ms / 1000.0 / args.speedup)
        if sem is not None:
            async with sem:
                res = await run_turn(session, args, layout, spec)
        else:
            res = await run_turn(session, args, layout, spec)
        results.append(res)
        if not res.ok and args.verbose:
            print(
                f"  [fail] conv={res.conv_id} turn={res.turn_idx}: {res.error}",
                file=sys.stderr,
            )


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


@dataclass
class Report:
    label: str
    graded: list[TurnResult] = field(default_factory=list)
    warmup: list[TurnResult] = field(default_factory=list)

    def render(self) -> str:
        ok = [r for r in self.graded if r.ok]
        failed = [r for r in self.graded if not r.ok]
        warm_failed = [r for r in self.warmup if not r.ok]
        ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
        tpots = [r.tpot_ms for r in ok if r.tpot_ms is not None]
        totals = [r.total_ms for r in ok if r.total_ms is not None]

        def fmt(v: float | None) -> str:
            return "n/a" if v is None else f"{v:8.1f}"

        lines = [
            "",
            "=" * 62,
            f"  {self.label}",
            "=" * 62,
            f"  graded requests      : {len(self.graded)}  "
            f"(warmup excluded: {len(self.warmup)})",
            f"  succeeded            : {len(ok)}",
            f"  FAILED (graded)      : {len(failed)}",
            f"  failed (warmup)      : {len(warm_failed)}",
            "",
            f"  TTFT  p50            : {fmt(pct(ttfts, 50))} ms",
            f"  TTFT  p95            : {fmt(pct(ttfts, 95))} ms",
            f"  TTFT  p99            : {fmt(pct(ttfts, 99))} ms",
            f"  TTFT  mean           : "
            f"{fmt(statistics.fmean(ttfts) if ttfts else None)} ms",
            "",
            f"  TPOT  p50            : {fmt(pct(tpots, 50))} ms",
            f"  TPOT  p95            : {fmt(pct(tpots, 95))} ms",
            "",
            f"  E2E   p50            : {fmt(pct(totals, 50))} ms",
            f"  E2E   p95            : {fmt(pct(totals, 95))} ms",
            "",
            f"  output tokens (mean) : "
            f"{statistics.fmean([r.output_tokens for r in ok]) if ok else 0:.1f}",
        ]
        if failed:
            lines.append("")
            lines.append("  SUCCESS CRITERION VIOLATED: failed_count != 0")
            seen: dict[str, int] = defaultdict(int)
            for r in failed:
                seen[(r.error or "unknown")[:120]] += 1
            for err, n in sorted(seen.items(), key=lambda kv: -kv[1])[:5]:
                lines.append(f"    {n:4d} x {err}")
        else:
            lines.append("")
            lines.append("  failed_count == 0  [OK]")
        lines.append("=" * 62)
        return "\n".join(lines)


async def fetch_metrics(base_url: str) -> dict[str, float]:
    """Scrape the few prefix-cache gauges we care about from /metrics."""
    wanted = ("gpu_prefix_cache_hit_rate", "gpu_prefix_cache_queries", "gpu_cache_usage")
    out: dict[str, float] = {}
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{base_url}/metrics", timeout=aiohttp.ClientTimeout(10)) as r:
                if r.status != 200:
                    return out
                for line in (await r.text()).splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    if any(w in line for w in wanted):
                        name, _, value = line.rpartition(" ")
                        try:
                            out[name.strip()] = float(value)
                        except ValueError:
                            pass
    except Exception:  # noqa: BLE001 - metrics are best-effort
        pass
    return out


async def main_async(args: argparse.Namespace) -> int:
    convs = load_trace(args.trace)
    total_turns = sum(len(v) for v in convs.values())
    layout = PromptLayout(
        shared_prefix_tokens=args.shared_prefix_tokens,
        conv_prefix_tokens=args.conv_prefix_tokens,
    )
    print(
        f"Loaded {total_turns} turns across {len(convs)} conversations from "
        f"{args.trace}"
    )
    print(
        f"Prompt layout: shared={layout.shared_chars} chars "
        f"({args.shared_prefix_tokens} tok), conv={layout.conv_chars} chars "
        f"({args.conv_prefix_tokens} tok), remainder = turn-local filler"
    )
    if args.speedup != 1.0:
        print(f"WARNING: --speedup {args.speedup} compresses the arrival schedule; "
              "latency numbers are not comparable to a real run.")

    before = await fetch_metrics(args.base_url)
    results: list[TurnResult] = []
    sem = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        t_zero = time.perf_counter()
        await asyncio.gather(
            *(
                run_conversation(session, args, layout, turns, t_zero, results, sem)
                for turns in convs.values()
            )
        )
        wall = time.perf_counter() - t_zero

    after = await fetch_metrics(args.base_url)

    report = Report(label=args.label)
    for r in results:
        (report.warmup if r.in_warmup else report.graded).append(r)
    print(report.render())
    print(f"  wall clock           : {wall:.1f} s")
    if before or after:
        print("\n  server metrics (before -> after):")
        for k in sorted(set(before) | set(after)):
            print(f"    {k}: {before.get(k, float('nan')):.4f} -> "
                  f"{after.get(k, float('nan')):.4f}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "label": args.label,
                    "wall_s": wall,
                    "graded": [r.__dict__ for r in report.graded],
                    "warmup": [r.__dict__ for r in report.warmup],
                    "metrics_before": before,
                    "metrics_after": after,
                },
                f,
                indent=2,
            )
        print(f"\n  wrote {args.json_out}")

    # Non-zero exit when the hard success criterion is violated.
    return 1 if any(not r.ok for r in report.graded) else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default="LFM2.5-1.2B-Instruct")
    p.add_argument(
        "--trace", default="../input_part2/trace_grading_public.jsonl",
        help="Path to trace_grading_public.jsonl",
    )
    p.add_argument("--label", default="baseline", help="Label for the report header")
    p.add_argument(
        "--shared-prefix-tokens", type=int, default=1000,
        help="Size of the system prefix shared by every request",
    )
    p.add_argument(
        "--conv-prefix-tokens", type=int, default=1000,
        help="Size of the prefix shared by a conversation's turns",
    )
    p.add_argument(
        "--speedup", type=float, default=1.0,
        help="Divide all wait times by this. Use for smoke tests only.",
    )
    p.add_argument(
        "--max-concurrency", type=int, default=0,
        help="Cap in-flight requests (0 = uncapped, matching the real trace)",
    )
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--json-out", default=None, help="Write raw results to this file")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(parse_args())))
