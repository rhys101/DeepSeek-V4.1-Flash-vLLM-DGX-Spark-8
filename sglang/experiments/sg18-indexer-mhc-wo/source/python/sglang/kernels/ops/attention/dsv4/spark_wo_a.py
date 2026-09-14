"""TP8 single-group BF16 WO-A with FP32 split-K accumulation."""
import torch
import triton
import triton.language as tl


@triton.jit
def _wo_partial(X, W, P, Y, M: tl.constexpr, SX: tl.constexpr,
                SPLITS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    mt, nt, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = mt * 16 + tl.arange(0, 16)
    n = nt * BN + tl.arange(0, BN)
    k = split * (4096 // SPLITS) + tl.arange(0, BK)
    acc = tl.zeros((16, BN), tl.float32)
    for i in range(4096 // SPLITS // BK):
        offsets = k + i * BK
        x = tl.load(X + m[:, None] * SX + offsets[None, :], m[:, None] < M, 0)
        w = tl.load(W + n[None, :] * 4096 + offsets[:, None])
        acc += tl.dot(x, w)
    if SPLITS == 1:
        tl.store(Y + m[:, None] * 1024 + n[None, :], acc, m[:, None] < M)
    else:
        tl.store(P + (split * M + m[:, None]) * 1024 + n[None, :], acc, m[:, None] < M)


@triton.jit
def _wo_reduce(P, Y, E: tl.constexpr, SPLITS: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    split = tl.arange(0, SPLITS)
    values = tl.load(P + split[:, None] * E + i[None, :], i[None, :] < E, 0)
    tl.store(Y + i, tl.sum(values, 0), i < E)


def wo_a_spark(x, weight, *, splits=4, bn=64, bk=128):
    m = x.shape[0]
    assert 0 < m <= 96 and x.shape[1:] == (1, 4096)
    assert weight.shape == (1, 1024, 4096) and weight.is_contiguous()
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and x.device == weight.device
    assert x.stride(2) == 1 and x.stride(0) >= 4096
    assert splits in (1, 2, 4, 8) and bn in (32, 64, 128) and bk in (64, 128)
    out = torch.empty((m, 1, 1024), dtype=x.dtype, device=x.device)
    partial = torch.empty((splits, m, 1024), dtype=torch.float32, device=x.device) if splits > 1 else out
    _wo_partial[(triton.cdiv(m, 16), 1024 // bn, splits)](
        x, weight, partial, out, m, x.stride(0), splits, bn, bk,
        num_warps=4, num_stages=3)
    if splits > 1:
        _wo_reduce[(triton.cdiv(m * 1024, 256),)](partial, out, m * 1024, splits, num_warps=4)
    return out


@triton.jit
def _wo_reduce_quant(P, Q, S, M: tl.constexpr, SPLITS: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    i = tile * 256 + tl.arange(0, 256)
    split = tl.arange(0, SPLITS)
    values = tl.load(P + split[:, None] * (M * 1024) + row * 1024 + i[None, :])
    # Preserve the deployed BF16 materialization before computing MXFP8 scales.
    y = tl.sum(values, 0).to(tl.bfloat16).to(tl.float32).reshape((8, 32))
    amax = tl.max(tl.abs(y), 1)
    normalized = amax * (1.0 / 448.0)
    bits = normalized.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = tl.where(normalized <= 0, 0, tl.minimum(exponent + bump.to(tl.int32), 254))
    # Match SGLang's explicitly selected FlashInfer CuTe quantizer, including
    # its byte-zero inverse convention. FlashInfer's default CUDA backend has
    # different behavior for tiny values and is not this input contract.
    inv = tl.where(sf == 0, 0, ((254 - sf) << 23)).to(tl.float32, bitcast=True)
    quant = tl.minimum(tl.maximum(y * inv[:, None], -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + row * 1024 + i, quant.reshape((256,)))
    col = tile * 8 + tl.arange(0, 8)
    offset = (col // 4) * 512 + (row % 32) * 16 + (row // 32) * 4 + col % 4
    tl.store(S + offset, sf.to(tl.uint8))
    # The consumer may inspect every padded scale byte. Valid and padding rows
    # have disjoint writers; no clearing kernel or cross-CTA ordering is needed.
    for z in tl.static_range(triton.cdiv(4096, M * 4 * 256)):
        s = (row * 4 + tile) * 256 + tl.arange(0, 256) + z * (M * 4 * 256)
        scale_row = (s % 512) // 16 + ((s % 16) // 4) * 32
        tl.store(S + s, 0, (s < 4096) & (scale_row >= M))


def quantize_partial_spark(partial):
    splits, m, n = partial.shape
    assert splits in (1, 2, 4, 8) and 0 < m <= 96 and n == 1024
    assert partial.is_contiguous() and partial.dtype == torch.float32
    q = torch.empty((m, 1024), device=partial.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(4096, device=partial.device, dtype=torch.uint8)
    _wo_reduce_quant[(m, 4)](partial, q, scales, m, splits, num_warps=4)
    return q, scales


def wo_a_spark_packed(x, weight, *, splits=2, bn=64):
    """Single-group WO-A followed by BF16 rounding and fused output quantization."""
    m = x.shape[0]
    assert 0 < m <= 16 and x.shape[1:] == (1, 4096)
    assert weight.shape == (1, 1024, 4096) and weight.is_contiguous()
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and x.device == weight.device and x.stride(2) == 1
    assert splits in (1, 2, 4, 8) and bn in (32, 64, 128)
    partial = torch.empty((splits, m, 1024), device=x.device, dtype=torch.float32)
    _wo_partial[(triton.cdiv(m, 16), 1024 // bn, splits)](
        x, weight, partial, partial, m, x.stride(0), splits, bn, 128,
        num_warps=4, num_stages=3)
    q, scales = quantize_partial_spark(partial)
    from sglang.srt.layers.quantization.spark_mxfp8_input import SparkMxfp8SwizzledInput
    return SparkMxfp8SwizzledInput(q, scales)
