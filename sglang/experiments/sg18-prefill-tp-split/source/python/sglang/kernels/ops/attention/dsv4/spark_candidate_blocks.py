"""SM121 bounded candidate mask work with graph-owned persistent tail validity.

The selector and candidate budget are unchanged. A workspace is owned by one
attention backend and one planned tensor shape/role. Its CUDA work must remain
ordered on that backend's execution stream. Initial allocation occurs during
uncaptured warmup, never during graph capture or replay.
"""
from functools import lru_cache
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _max_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _bounded_scores(X, LENS, OUT, SCORES, OLD_OUT, OLD_SCORE,
                    WIDTH: tl.constexpr, STRIDE: tl.constexpr,
                    BLOCKS: tl.constexpr, GROUP: tl.constexpr,
                    GROUP_PAD: tl.constexpr, WORKERS: tl.constexpr,
                    TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    worker = tl.program_id(1)
    logical_length = tl.load(LENS + row)
    length = tl.minimum(tl.maximum(logical_length, 0), WIDTH)
    dirty = tl.maximum(tl.load(OLD_OUT + row), tl.load(OLD_SCORE + row))
    extent = tl.maximum(length, dirty)
    end_block = tl.cdiv(extent, GROUP)
    first = worker * TILE
    while first < end_block:
        blocks = first + tl.arange(0, TILE)
        offsets = tl.arange(0, GROUP_PAD)
        cols = blocks[:, None].to(tl.int64) * GROUP + offsets[None, :]
        valid = (cols < WIDTH) & (offsets[None, :] < GROUP)
        values = tl.load(X + row * STRIDE + cols,
                         valid & (cols < length), other=-float('inf')).to(tl.float32)
        # Write whole touched blocks, including a partially visible last block.
        tl.store(OUT + row * WIDTH + cols, values, valid & (blocks[:, None] < end_block))
        scores = tl.reduce(values, axis=1, combine_fn=_max_nan)
        # Clamp physical work, while preserving the reference's newest-block
        # rule if logical metadata extends beyond this tensor's width.
        scores = tl.where((logical_length > 0) & (blocks == (logical_length - 1) // GROUP), float('inf'), scores)
        tl.store(SCORES + row * BLOCKS + blocks, scores, (blocks < BLOCKS) & (blocks < end_block))
        first += WORKERS * TILE


@triton.jit
def _bounded_mask(X, LENS, KEEP, OUT, OLD_OUT,
                  WIDTH: tl.constexpr, STRIDE: tl.constexpr,
                  KEEP_STRIDE: tl.constexpr, KEEP_COL_STRIDE: tl.constexpr,
                  WORKERS: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    worker = tl.program_id(1)
    length = tl.minimum(tl.maximum(tl.load(LENS + row), 0), WIDTH)
    extent = tl.maximum(length, tl.load(OLD_OUT + row))
    first = worker * TILE
    while first < extent:
        cols = first + tl.arange(0, TILE).to(tl.int64)
        visible = (cols < WIDTH) & (cols < length)
        keep = tl.load(KEEP + row * KEEP_STRIDE + cols * KEEP_COL_STRIDE, visible, other=0)
        values = tl.load(X + row * STRIDE + cols, visible & keep, other=-float('inf')).to(tl.float32)
        tl.store(OUT + row * WIDTH + cols, values, (cols < WIDTH) & (cols < extent))
        first += WORKERS * TILE


@triton.jit
def _advance_extents(LENS, OLD_OUT, OLD_SCORE, ROWS: tl.constexpr,
                     WIDTH: tl.constexpr, SOURCE: tl.constexpr, TILE: tl.constexpr):
    row = tl.arange(0, TILE)
    length = tl.load(LENS + row, row < ROWS, 0)
    length = tl.minimum(tl.maximum(length, 0), WIDTH)
    tl.store(OLD_OUT + row, length, row < ROWS)
    if SOURCE:
        tl.store(OLD_SCORE + row, length, row < ROWS)


@triton.jit
def _clear_previous(OLD_IDS, KEEP, WIDTH: tl.constexpr, GROUP: tl.constexpr,
                     TOPK: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * TILE + tl.arange(0, TILE)
    block = tl.load(OLD_IDS + row * TOPK + i // GROUP, i < TOPK * GROUP, -1).to(tl.int64)
    col = block * GROUP + i % GROUP
    tl.store(KEEP + row * WIDTH + col, False,
             (i < TOPK * GROUP) & (block >= 0) & (col < WIDTH))


@triton.jit
def _publish_and_remember(IDS, VALUES, KEEP, OLD_IDS,
                          WIDTH: tl.constexpr, GROUP: tl.constexpr,
                          TOPK: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * TILE + tl.arange(0, TILE)
    selected = tl.load(IDS + row * TOPK + i // GROUP, i < TOPK * GROUP, 0).to(tl.int64)
    score = tl.load(VALUES + row * TOPK + i // GROUP, i < TOPK * GROUP, -float('inf'))
    col = selected * GROUP + i % GROUP
    tl.store(KEEP + row * WIDTH + col, score > -float('inf'),
             (i < TOPK * GROUP) & (col < WIDTH))
    tl.store(OLD_IDS + row * TOPK + i // GROUP, selected,
             (i < TOPK * GROUP) & (i % GROUP == 0))


class CandidateWorkspace:
    def __init__(self, rows, width, group, topk, device, source):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Bounded indexer workspace must be initialized before capture')
        self.rows, self.width, self.group = rows, width, group
        self.source = source
        self.blocks = triton.cdiv(width, group)
        self.topk = min(topk, self.blocks)
        self.output = torch.full((rows, width), -torch.inf, device=device, dtype=torch.float32)
        self.old_output = torch.zeros(rows, device=device, dtype=torch.int64)
        self.old_score = torch.zeros(rows, device=device, dtype=torch.int64)
        if source:
            self.scores = torch.full((rows, self.blocks), -torch.inf, device=device, dtype=torch.float32)
            self.keep = torch.zeros((rows, width), device=device, dtype=torch.bool)
            self.old_ids = torch.full((rows, self.topk), -1, device=device, dtype=torch.int64)

    def run(self, logits, lengths, published, workers=4):
        rows, width = self.rows, self.width
        if self.source:
            assert published is None
            _bounded_scores[(rows, workers)](
                logits, lengths, self.output, self.scores, self.old_output, self.old_score,
                width, logits.stride(0), self.blocks, self.group,
                triton.next_power_of_2(self.group), workers, 128, num_warps=4)
            _advance_extents[(1,)](lengths, self.old_output, self.old_score,
                                  rows, width, True, triton.next_power_of_2(rows), num_warps=4)
            # Fixed tensor shape and the deployed selector are deliberately retained.
            selected = self.scores.topk(self.topk, dim=-1, sorted=False)
            grid = (rows, triton.cdiv(self.topk * self.group, 256))
            _clear_previous[grid](self.old_ids, self.keep, width, self.group, self.topk, 256, num_warps=4)
            _publish_and_remember[grid](selected.indices, selected.values, self.keep,
                                        self.old_ids, width, self.group, self.topk, 256, num_warps=4)
            return self.output, self.keep
        assert published is not None
        _bounded_mask[(rows, workers)](
            logits, lengths, published, self.output, self.old_output, width, logits.stride(0),
            published.stride(0), published.stride(1), workers, 4096, num_warps=4)
        _advance_extents[(1,)](lengths, self.old_output, self.old_score,
                              rows, width, False, triton.next_power_of_2(rows), num_warps=4)
        return self.output, None


@lru_cache(maxsize=1)
def enabled_on_device():
    return os.environ.get('SPARK_KERNEL_INDEXER', '0') == '1' and torch.cuda.get_device_capability() == (12, 1)


def candidate_block_logits_bounded(logits, seq_lens, *, topk_blocks, block_size, published, owner):
    from sglang.kernels.ops.attention.dsv4.candidate_blocks import candidate_block_logits
    rows, width = logits.shape
    # Initial qualification is deliberately confined to the two target graph shapes.
    if not enabled_on_device() or owner is None or rows not in (6, 48):
        return candidate_block_logits(logits, seq_lens, topk_blocks=topk_blocks,
                                      block_size=block_size, published=published)
    source = published is None
    key = (rows, width, block_size, topk_blocks, source, logits.device.index)
    if not hasattr(owner, '_spark_candidate_workspaces'):
        owner._spark_candidate_workspaces = {}
    workspaces = owner._spark_candidate_workspaces
    if key not in workspaces:
        workspaces[key] = CandidateWorkspace(rows, width, block_size, topk_blocks, logits.device, source)
    workers = int(os.environ.get('SPARK_INDEXER_WORKERS') or ('16' if rows <= 8 else '4'))
    return workspaces[key].run(logits, seq_lens, published, workers=workers)
