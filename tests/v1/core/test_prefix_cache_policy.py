# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the configurable prefix-cache eviction policy.

Covers the segmented free list (SLRU), the depth-based classification in
`BlockPool.free_blocks`, the filler-hashing skip, and CLI/env config resolution.
"""

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.prefix_cache_policy import (
    PrefixCacheEvictionStats,
    PrefixCachePolicyConfig,
)

BLOCK_SIZE = 16


def make_blocks(n: int) -> list[KVCacheBlock]:
    return [KVCacheBlock(block_id=i) for i in range(n)]


def ids(blocks) -> list[int]:
    return [b.block_id for b in blocks]


def cache_block(
    pool: BlockPool, block: KVCacheBlock, depth_tokens: int, group_id: int = 0
) -> None:
    """Cache `block` at `depth_tokens` of prefix depth.

    Goes through `_insert_block_hash` so the block lands in
    `cached_block_hash_to_block` too — setting `block_hash` alone would leave the
    map empty and `_maybe_evict_cached_block` would then report no eviction.
    The hash embeds the block id so distinct blocks get distinct keys.
    """
    block_hash = make_block_hash_with_group_id(
        BlockHash(f"h{block.block_id}".encode()), group_id
    )
    pool._insert_block_hash(block_hash, block, num_tokens=depth_tokens)


def contains(blocks, target: KVCacheBlock) -> bool:
    """Identity-based membership.

    `KVCacheBlock` is a dataclass, so `in` would use the generated `__eq__`,
    which compares the `prev_free_block`/`next_free_block` links and recurses.
    """
    return any(b is target for b in blocks)


# --------------------------------------------------------------- free list


def test_lru_mode_uses_a_single_segment():
    """With nothing marked protected, behavior matches the original LRU list."""
    q = FreeKVCacheBlockQueue(make_blocks(5))
    assert q.num_free_blocks == 5
    assert q.num_blocks_in_segment(protected=True) == 0
    assert ids(q.popleft_n(5)) == [0, 1, 2, 3, 4]
    assert q.num_free_blocks == 0


def test_probationary_drains_before_protected():
    blocks = make_blocks(6)
    q = FreeKVCacheBlockQueue(blocks)
    popped = q.popleft_n(6)
    q.append_n(popped[0:2], protected=True)  # 0, 1 protected
    q.append_n(popped[2:4])  # 2, 3 probationary

    assert q.num_free_blocks == 4
    assert q.num_blocks_in_segment(protected=True) == 2
    assert q.num_blocks_in_segment(protected=False) == 2
    # Probationary victims first, protected only once probation is exhausted.
    assert ids(q.popleft_n(4)) == [2, 3, 0, 1]


def test_popleft_n_spans_the_segment_boundary():
    blocks = make_blocks(6)
    q = FreeKVCacheBlockQueue(blocks)
    popped = q.popleft_n(6)
    q.append_n(popped[0:2], protected=True)
    q.append_n(popped[2:5])

    assert ids(q.popleft_n(4)) == [2, 3, 4, 0]
    assert q.num_free_blocks == 1
    assert ids(q.get_all_free_blocks()) == [1]


def test_popleft_crosses_into_protected_and_then_raises():
    blocks = make_blocks(3)
    q = FreeKVCacheBlockQueue(blocks)
    popped = q.popleft_n(3)
    q.append_n([popped[0]], protected=True)

    assert q.popleft().block_id == 0
    assert q.num_free_blocks == 0
    with pytest.raises(ValueError, match="No free blocks available"):
        q.popleft()


def test_remove_is_segment_agnostic():
    """`remove` unlinks via the block's own neighbours, so it works in both."""
    blocks = make_blocks(5)
    q = FreeKVCacheBlockQueue(blocks)
    popped = q.popleft_n(5)
    q.append_n(popped[0:3], protected=True)
    q.append_n(popped[3:5])

    q.remove(popped[1])  # middle of the protected segment
    assert q.num_free_blocks == 4
    assert ids(q.get_all_free_blocks()) == [3, 4, 0, 2]

    q.remove(popped[3])  # head of the probationary segment
    assert q.num_free_blocks == 3
    assert ids(q.get_all_free_blocks()) == [4, 0, 2]


def test_prepend_n_targets_the_requested_segment():
    blocks = make_blocks(4)
    q = FreeKVCacheBlockQueue(blocks)
    popped = q.popleft_n(4)
    q.append_n([popped[0], popped[1]], protected=True)
    q.prepend_n([popped[2]], protected=True)
    assert ids(q.get_all_free_blocks()) == [2, 0, 1]


def test_iter_blocks_after_spans_both_segments():
    blocks = make_blocks(6)
    q = FreeKVCacheBlockQueue(blocks)
    popped = q.popleft_n(6)
    q.append_n(popped[0:2], protected=True)
    q.append_n(popped[2:5])

    assert ids(list(q.iter_blocks_after(None))) == [2, 3, 4, 0, 1]
    # Crossing the boundary from the last probationary block.
    assert ids(list(q.iter_blocks_after(popped[4]))) == [0, 1]
    assert list(q.iter_blocks_after(popped[1])) == []
    # Iteration order must match the order popleft would hand blocks out.
    assert ids(list(q.iter_blocks_after(None))) == ids(q.popleft_n(5))


def test_empty_queue():
    q = FreeKVCacheBlockQueue([])
    assert q.num_free_blocks == 0
    assert q.get_all_free_blocks() == []
    assert list(q.iter_blocks_after(None)) == []


# --------------------------------------------------------------- block pool


def make_pool(**policy_kwargs) -> BlockPool:
    policy = PrefixCachePolicyConfig(**policy_kwargs)
    return BlockPool(
        num_gpu_blocks=64,
        enable_caching=True,
        hash_block_size=BLOCK_SIZE,
        policy_config=policy,
    )


def test_free_blocks_lru_keeps_everything_probationary():
    pool = make_pool(policy="lru")
    blocks = pool.get_new_blocks(4)
    for i, b in enumerate(blocks):
        cache_block(pool, b, depth_tokens=(i + 1) * BLOCK_SIZE)
    pool.free_blocks(blocks)
    assert pool.free_block_queue.num_blocks_in_segment(protected=True) == 0


def test_free_blocks_slru_splits_by_depth():
    pool = make_pool(policy="slru", slru_protected_tokens=1000)
    blocks = pool.get_new_blocks(4)
    cache_block(pool, blocks[0], depth_tokens=500)  # protected
    cache_block(pool, blocks[1], depth_tokens=1000)  # protected (boundary inclusive)
    cache_block(pool, blocks[2], depth_tokens=1001)  # probationary
    cache_block(pool, blocks[3], depth_tokens=3000)  # probationary

    pool.free_blocks(blocks)
    q = pool.free_block_queue
    assert q.num_blocks_in_segment(protected=True) == 2
    protected_ids = set(ids(q.get_all_free_blocks())[-2:])
    assert protected_ids == {blocks[0].block_id, blocks[1].block_id}


def test_free_blocks_unhashed_still_die_first_under_slru():
    pool = make_pool(policy="slru", slru_protected_tokens=1000)
    blocks = pool.get_new_blocks(3)
    cache_block(pool, blocks[0], depth_tokens=500)  # protected
    cache_block(pool, blocks[1], depth_tokens=5000)  # probationary
    # blocks[2] stays unhashed -> must be reclaimed before anything cached.
    pool.free_blocks(blocks)

    order = ids(pool.free_block_queue.get_all_free_blocks())
    assert order[0] == blocks[2].block_id
    assert order[-1] == blocks[0].block_id


def test_slru_allocation_prefers_probationary_victims():
    """A shallow (protected) block survives an allocation that a deep one does not."""
    pool = make_pool(policy="slru", slru_protected_tokens=1000)
    free_before = pool.get_num_free_blocks()
    blocks = pool.get_new_blocks(2)
    shallow, deep = blocks
    cache_block(pool, shallow, depth_tokens=100)
    cache_block(pool, deep, depth_tokens=9000)
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == free_before

    # Drain everything except one block: the protected shallow block must be
    # the very last one standing.
    remaining = pool.get_new_blocks(pool.get_num_free_blocks() - 1)
    assert not contains(remaining, shallow)
    assert contains(remaining, deep)


# ------------------------------------------------------------- filler skip


@pytest.mark.parametrize(
    "enabled,depth,num_tokens,expected",
    [
        (True, 2000, 1984, False),  # inside the protected prefix
        (True, 2000, 2000, False),  # exactly at the boundary
        (True, 2000, 2016, True),  # beyond -> filler
        (False, 2000, 9999, False),  # disabled -> never filler
    ],
)
def test_is_filler_depth(enabled, depth, num_tokens, expected):
    pool = make_pool(enable_skip_filler_hashing=enabled, filler_depth_tokens=depth)
    assert pool._is_filler_depth(num_tokens) is expected


def test_filler_skip_is_disabled_when_caching_is_off():
    pool = BlockPool(
        num_gpu_blocks=16,
        enable_caching=False,
        hash_block_size=BLOCK_SIZE,
        policy_config=PrefixCachePolicyConfig(enable_skip_filler_hashing=True),
    )
    assert pool.skip_filler_hashing is False


# ------------------------------------------------------------------ config


def test_defaults(monkeypatch):
    for k in (
        "VLLM_PREFIX_CACHE_POLICY",
        "VLLM_SKIP_FILLER_HASHING",
        "VLLM_FILLER_DEPTH_TOKENS",
        "VLLM_SLRU_PROTECTED_TOKENS",
        "VLLM_PREFIX_CACHE_STATS",
    ):
        monkeypatch.delenv(k, raising=False)
    cfg = PrefixCachePolicyConfig.from_config(None)
    assert cfg.policy == "lru"
    assert cfg.use_slru is False
    assert cfg.enable_skip_filler_hashing is True
    assert cfg.filler_depth_tokens == 2000
    assert cfg.slru_protected_tokens == 1000
    assert cfg.collect_stats is False


def test_env_vars_are_honored(monkeypatch):
    monkeypatch.setenv("VLLM_PREFIX_CACHE_POLICY", "slru")
    monkeypatch.setenv("VLLM_SKIP_FILLER_HASHING", "0")
    monkeypatch.setenv("VLLM_FILLER_DEPTH_TOKENS", "3072")
    monkeypatch.setenv("VLLM_SLRU_PROTECTED_TOKENS", "1536")
    monkeypatch.setenv("VLLM_PREFIX_CACHE_STATS", "true")
    cfg = PrefixCachePolicyConfig.from_config(None)
    assert cfg.use_slru
    assert cfg.enable_skip_filler_hashing is False
    assert cfg.filler_depth_tokens == 3072
    assert cfg.slru_protected_tokens == 1536
    assert cfg.collect_stats is True


def test_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("VLLM_PREFIX_CACHE_POLICY", "slru")
    monkeypatch.setenv("VLLM_FILLER_DEPTH_TOKENS", "3072")

    class FakeCacheConfig:
        prefix_cache_policy = "lru"
        enable_skip_filler_hashing = False
        filler_depth_tokens = 512
        slru_protected_tokens = 256
        prefix_cache_stats = True

    cfg = PrefixCachePolicyConfig.from_config(FakeCacheConfig())
    assert cfg.policy == "lru"
    assert cfg.enable_skip_filler_hashing is False
    assert cfg.filler_depth_tokens == 512
    assert cfg.slru_protected_tokens == 256
    assert cfg.collect_stats is True


def test_unset_cli_falls_through_to_env(monkeypatch):
    monkeypatch.setenv("VLLM_PREFIX_CACHE_POLICY", "slru")
    monkeypatch.setenv("VLLM_FILLER_DEPTH_TOKENS", "3072")

    class FakeCacheConfig:
        prefix_cache_policy = None
        enable_skip_filler_hashing = None
        filler_depth_tokens = None
        slru_protected_tokens = None
        prefix_cache_stats = None

    cfg = PrefixCachePolicyConfig.from_config(FakeCacheConfig())
    assert cfg.policy == "slru"
    assert cfg.filler_depth_tokens == 3072


@pytest.mark.parametrize(
    "var,value,match",
    [
        ("VLLM_PREFIX_CACHE_POLICY", "lfu", "not a valid prefix cache"),
        ("VLLM_FILLER_DEPTH_TOKENS", "abc", "not a valid integer"),
        ("VLLM_SKIP_FILLER_HASHING", "maybe", "not a valid boolean"),
    ],
)
def test_invalid_values_are_rejected(monkeypatch, var, value, match):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError, match=match):
        PrefixCachePolicyConfig.from_config(None)


# ------------------------------------------------------------------- stats


def test_stats_counters():
    stats = PrefixCacheEvictionStats()
    stats.record_allocation(5, 100)
    stats.record_allocation(3, 40)
    stats.record_eviction(1024)
    stats.record_eviction(None)
    stats.record_touch(7)

    assert stats.allocations == 8
    assert stats.min_free_blocks == 40
    assert stats.evictions == 2
    assert stats.touches == 7
    assert "evictions=2" in stats.summary()


def test_stats_are_off_by_default():
    pool = make_pool()
    assert pool.policy_stats is None


def test_stats_record_evictions_when_enabled():
    pool = make_pool(collect_stats=True)
    blocks = pool.get_new_blocks(2)
    for i, b in enumerate(blocks):
        cache_block(pool, b, depth_tokens=(i + 1) * BLOCK_SIZE)
    pool.free_blocks(blocks)
    assert pool.policy_stats is not None
    assert pool.policy_stats.allocations == 2

    # Drain the pool so the freed, still-cached blocks get reallocated.
    pool.get_new_blocks(pool.get_num_free_blocks())
    assert pool.policy_stats.evictions >= 2
