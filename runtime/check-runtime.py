"""Exercise actual metadata construction, indexer scheduling and top-k on GB10."""
from types import SimpleNamespace as NS
import torch
from vllm.models.deepseek_v4_1.attention import DeepseekV4IndexerCache
from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import DeepseekSparseSWAFlashInferMetadataBuilder
from vllm.v1.attention.backends.mla.sparse_swa import _COMPUTE_SWA_INDICES_AND_LENS_KERNEL
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec
from vllm.v1.worker.utils import select_common_block_size
from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata
import vllm._custom_ops as ops

torch.set_num_threads(2)
torch.manual_seed(41)
assert torch.cuda.get_device_capability() == (12, 1)
spec = SlidingWindowMLASpec(block_size=64, num_kv_heads=1, head_size=512,
                           dtype=torch.uint8, sliding_window=128)
for text_only, expected_width in ((False, 1152),):
    cfg = NS(model_config=NS(max_model_len=8192,
                            hf_config=NS(sliding_window=128, vision_n_layers=27,
                                         vision_max_n_token=1024, compress_ratios=[0, 1, 2]),
                            multimodal_config=NS(language_model_only=text_only)),
             scheduler_config=NS(max_num_batched_tokens=1024, max_num_seqs=1),
             speculative_config=None)
    builder = DeepseekSparseSWAFlashInferMetadataBuilder(spec, ['test'], cfg, torch.device('cuda'))
    assert builder.prefill_index_width == expected_width
    assert builder.prefill_swa_indices.shape == (1024, 1, expected_width)
    print(f'PASS: text_only={text_only} metadata width={expected_width}', flush=True)
    if not text_only:
        continue
    for tokens, prefix in ((16, 0), (80, 32), (80, 200), (1024, 1000)):
        indices = builder.prefill_swa_indices[:tokens]
        lengths = builder.prefill_swa_lens[:tokens]
        starts = torch.tensor([0, tokens], dtype=torch.int32, device='cuda')
        seq_lens = torch.tensor([prefix + tokens], dtype=torch.int32, device='cuda')
        reqs = torch.zeros(tokens, dtype=torch.int32, device='cuda')
        valid = torch.ones(tokens, dtype=torch.bool, device='cuda')
        blocks = torch.arange(40, dtype=torch.int32, device='cuda').flip(0).unsqueeze(0)
        _COMPUTE_SWA_INDICES_AND_LENS_KERNEL(
            indices, lengths, 128, 128, lengths, lengths,
            starts, seq_lens, reqs, valid, blocks, 64,
            num_tokens=tokens, token_offset=0, has_image=False)
        reference = torch.full_like(indices, -1)
        expected_lens = []
        for t in range(tokens):
            positions = torch.arange(max(0, prefix + t - 127), prefix + t + 1, device='cuda')
            expected_lens.append(len(positions))
            reference[t, 0, :len(positions)] = blocks[0, positions // 64] * 64 + positions % 64
        torch.testing.assert_close(indices, reference, atol=0, rtol=0)
        torch.testing.assert_close(lengths, torch.tensor(expected_lens, device='cuda', dtype=torch.int32), atol=0, rtol=0)
        print(f'PASS: exact causal SWA indices tokens={tokens} prefix={prefix}', flush=True)

for ratio in (1, 2):
    cache = DeepseekV4IndexerCache.__new__(DeepseekV4IndexerCache)
    torch.nn.Module.__init__(cache)
    cache.cache_config = NS(block_size=128)
    cache.head_dim, cache.dtype, cache.compress_ratio = 132, torch.uint8, ratio
    index_spec = cache.get_kv_cache_spec(NS(cache_config=NS(cache_dtype='fp8_ds_mla')))
    assert index_spec.num_states == 64
    assert select_common_block_size(index_spec.block_size, [cache.get_attn_backend()]) == index_spec.block_size
    scheduling = get_paged_mqa_logits_metadata(
        torch.tensor([[1000]], device='cuda', dtype=torch.int32), index_spec.num_states,
        torch.cuda.get_device_properties(0).multi_processor_count)
    torch.cuda.synchronize()
    print(f'PASS: indexer ratio={ratio} states={index_spec.num_states} scheduler={tuple(scheduling.shape)}', flush=True)

for width in (600, 4096, 8192, 32768):
    logits = torch.randn(2, width, device='cuda', dtype=torch.float32)
    lengths = torch.full((2, 1), width, device='cuda', dtype=torch.int32)
    result = torch.full((2, 512), -1, device='cuda', dtype=torch.int32)
    ops.top_k_per_row_decode(logits, 1, lengths, result, 2, logits.stride(0), logits.stride(1), 512)
    reference = torch.topk(logits, 512, dim=1).indices
    for i in range(2):
        assert set(result[i].tolist()) == set(reference[i].tolist())
    print(f'PASS: per-row top-k matches torch.topk width={width}', flush=True)
print('PASS: vision metadata and indexer regression checks', flush=True)
