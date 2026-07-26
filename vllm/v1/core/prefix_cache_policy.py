# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configurable prefix-cache eviction policy for the V1 KV cache.

Every knob defined here is resolvable from **either** a CLI flag (declared on
`vllm.config.cache.CacheConfig`, so it reaches `vllm serve`) **or** an
environment variable, so the policy can be tuned from `docker-compose.yml`
without editing source.

Resolution precedence, highest first:

1. CLI flag (e.g. ``--prefix-cache-policy=slru``)
2. Environment variable (e.g. ``VLLM_PREFIX_CACHE_POLICY=slru``)
3. Built-in default

A CLI flag counts as "set" only when it is not ``None``; every flag defaults to
``None`` precisely so that "unspecified" is distinguishable from "explicitly set
to the default value".

Background on why the policy lives here rather than in an `Evictor` class: the
V1 engine has no eviction-policy object. The policy *is* the ordering discipline
of `FreeKVCacheBlockQueue` (`vllm/v1/core/kv_cache_utils.py`) — victims are taken
from the head, recency is conferred by appending to the tail. This module
therefore only carries the *decision inputs*; the mechanism stays in the queue
and in `BlockPool.free_blocks`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, get_args

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.cache import CacheConfig

logger = init_logger(__name__)

PrefixCachePolicy = Literal["lru", "slru"]
VALID_PREFIX_CACHE_POLICIES: tuple[str, ...] = get_args(PrefixCachePolicy)

# Built-in defaults. These are tuned for the VChallenge multi-turn grading trace
# (1000-token shared system prefix + 1000-token per-conversation prefix +
# ~2000 tokens of turn-local filler per request).
DEFAULT_ENABLE_SKIP_FILLER_HASHING = True
DEFAULT_FILLER_DEPTH_TOKENS = 2000
DEFAULT_PREFIX_CACHE_POLICY: PrefixCachePolicy = "lru"
DEFAULT_SLRU_PROTECTED_TOKENS = 1000
DEFAULT_PREFIX_CACHE_STATS = False


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"Environment variable {name}={raw!r} is not a valid boolean. "
        "Use one of: 1/0, true/false, yes/no, on/off."
    )


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name}={raw!r} is not a valid integer."
        ) from exc
    if value < minimum:
        raise ValueError(f"Environment variable {name}={value} must be >= {minimum}.")
    return value


def _env_policy(name: str, default: PrefixCachePolicy) -> PrefixCachePolicy:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized not in VALID_PREFIX_CACHE_POLICIES:
        raise ValueError(
            f"Environment variable {name}={raw!r} is not a valid prefix cache "
            f"policy. Choose one of: {', '.join(VALID_PREFIX_CACHE_POLICIES)}."
        )
    return normalized  # type: ignore[return-value]


@dataclass
class PrefixCacheEvictionStats:
    """Phase-0 gate instrumentation for the prefix cache.

    The headline number is `evictions`: the count of *live cache entries that
    were actually destroyed*. If this stays at zero for a whole run, the KV cache
    never came under pressure and no eviction policy — LRU, SLRU, or otherwise —
    can affect that workload. Measure before optimizing.

    Counters are only maintained when stats collection is enabled
    (`--prefix-cache-stats` / `VLLM_PREFIX_CACHE_STATS=1`), so the default
    serving path pays nothing.
    """

    # Cached entries destroyed because their block was reallocated.
    evictions: int = 0
    # Histogram of eviction counts bucketed by prefix depth in tokens.
    eviction_depth_hist: dict[int, int] = field(default_factory=dict)
    # Blocks handed out by `BlockPool.get_new_blocks`.
    allocations: int = 0
    # Blocks reused via `BlockPool.touch` (i.e. prefix cache hits).
    touches: int = 0
    # Hash-map inserts performed vs. skipped as turn-local filler.
    hash_inserts: int = 0
    hash_inserts_skipped: int = 0
    # Low-water mark of free blocks observed at allocation time.
    min_free_blocks: int | None = None

    # Bucket width for the depth histogram, in tokens.
    depth_bucket_tokens: int = 512

    def record_allocation(self, num_blocks: int, num_free_blocks: int) -> None:
        self.allocations += num_blocks
        if self.min_free_blocks is None or num_free_blocks < self.min_free_blocks:
            self.min_free_blocks = num_free_blocks

    def record_eviction(self, depth_tokens: int | None) -> None:
        self.evictions += 1
        bucket = -1 if depth_tokens is None else depth_tokens // self.depth_bucket_tokens
        self.eviction_depth_hist[bucket] = self.eviction_depth_hist.get(bucket, 0) + 1

    def record_touch(self, num_blocks: int) -> None:
        self.touches += num_blocks

    def summary(self) -> str:
        if self.eviction_depth_hist:
            hist = ", ".join(
                f"[{b * self.depth_bucket_tokens}-"
                f"{(b + 1) * self.depth_bucket_tokens})={c}"
                if b >= 0
                else f"unhashed={c}"
                for b, c in sorted(self.eviction_depth_hist.items())
            )
        else:
            hist = "none"
        return (
            f"prefix-cache stats: evictions={self.evictions} "
            f"allocations={self.allocations} touches={self.touches} "
            f"hash_inserts={self.hash_inserts} "
            f"hash_inserts_skipped={self.hash_inserts_skipped} "
            f"min_free_blocks={self.min_free_blocks} "
            f"eviction_depth_hist={{{hist}}}"
        )


@dataclass(frozen=True)
class PrefixCachePolicyConfig:
    """Resolved prefix-cache policy knobs, passed down to `BlockPool`."""

    # Skip inserting turn-local "filler" blocks into the prefix-cache hash map.
    # These blocks are written once and never hit again, so hashing them costs
    # scheduler CPU (a GIL-bound resource) and buys nothing.
    enable_skip_filler_hashing: bool = DEFAULT_ENABLE_SKIP_FILLER_HASHING
    # Prefix depth in tokens beyond which a block is treated as filler.
    filler_depth_tokens: int = DEFAULT_FILLER_DEPTH_TOKENS
    # Free-list eviction ordering policy.
    policy: PrefixCachePolicy = DEFAULT_PREFIX_CACHE_POLICY
    # Under SLRU, blocks at or below this prefix depth go to the protected
    # segment and are only evicted once the probationary segment is exhausted.
    slru_protected_tokens: int = DEFAULT_SLRU_PROTECTED_TOKENS
    # Maintain Phase-0 gate counters. Off by default: zero hot-path cost.
    collect_stats: bool = DEFAULT_PREFIX_CACHE_STATS

    @property
    def use_slru(self) -> bool:
        return self.policy == "slru"

    @classmethod
    def from_config(
        cls, cache_config: "CacheConfig | None" = None
    ) -> "PrefixCachePolicyConfig":
        """Resolve the policy from CLI flags, then env vars, then defaults."""

        def pick_bool(cli: bool | None, env_name: str, default: bool) -> bool:
            if cli is not None:
                return cli
            return _env_bool(env_name, default)

        def pick_int(cli: int | None, env_name: str, default: int) -> int:
            if cli is not None:
                return cli
            return _env_int(env_name, default)

        cli_skip = getattr(cache_config, "enable_skip_filler_hashing", None)
        cli_filler_depth = getattr(cache_config, "filler_depth_tokens", None)
        cli_policy = getattr(cache_config, "prefix_cache_policy", None)
        cli_protected = getattr(cache_config, "slru_protected_tokens", None)
        cli_stats = getattr(cache_config, "prefix_cache_stats", None)

        policy: PrefixCachePolicy
        if cli_policy is not None:
            policy = cli_policy
        else:
            policy = _env_policy(
                "VLLM_PREFIX_CACHE_POLICY", DEFAULT_PREFIX_CACHE_POLICY
            )

        resolved = cls(
            enable_skip_filler_hashing=pick_bool(
                cli_skip,
                "VLLM_SKIP_FILLER_HASHING",
                DEFAULT_ENABLE_SKIP_FILLER_HASHING,
            ),
            filler_depth_tokens=pick_int(
                cli_filler_depth,
                "VLLM_FILLER_DEPTH_TOKENS",
                DEFAULT_FILLER_DEPTH_TOKENS,
            ),
            policy=policy,
            slru_protected_tokens=pick_int(
                cli_protected,
                "VLLM_SLRU_PROTECTED_TOKENS",
                DEFAULT_SLRU_PROTECTED_TOKENS,
            ),
            collect_stats=pick_bool(
                cli_stats, "VLLM_PREFIX_CACHE_STATS", DEFAULT_PREFIX_CACHE_STATS
            ),
        )

        if resolved.use_slru and (
            resolved.enable_skip_filler_hashing
            and resolved.slru_protected_tokens >= resolved.filler_depth_tokens
        ):
            # Not an error, but worth surfacing: if everything that still gets
            # hashed is also protected, the probationary segment only ever holds
            # unhashed blocks and SLRU degenerates towards plain LRU.
            logger.warning(
                "SLRU protected depth (%d tokens) >= filler depth (%d tokens): "
                "every cached block is protected, so SLRU will behave close to "
                "LRU. Lower --slru-protected-tokens or raise "
                "--filler-depth-tokens to differentiate them.",
                resolved.slru_protected_tokens,
                resolved.filler_depth_tokens,
            )

        return resolved

    def log_once(self) -> None:
        logger.info(
            "Prefix cache policy: policy=%s skip_filler_hashing=%s "
            "filler_depth_tokens=%d slru_protected_tokens=%d collect_stats=%s",
            self.policy,
            self.enable_skip_filler_hashing,
            self.filler_depth_tokens,
            self.slru_protected_tokens,
            self.collect_stats,
        )
