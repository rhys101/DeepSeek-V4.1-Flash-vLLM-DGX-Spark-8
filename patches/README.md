# Reference attention patch

The build applies `reference-attention.patch` to the pristine pinned vLLM feature-source tree. [manifest.json](manifest.json) records all four before/after hashes; the build rejects unexpected source bytes and applies with zero fuzz.

The patch reproduces four files from [Tony’s pinned recipe](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/ca662ac35193c69ace9cee37f13a94abf2eff0fc): `attention.py`, `nvidia/flashinfer_sparse.py`, `sparse_swa.py` and `sparse_attn_indexer.py`.

- Physical SWA and compressed pages contain 64 states, for both compression ratios.
- The indexer scheduler uses 64-state pages and SM12x uses the per-row top-k path.
- Image metadata reserves the 1,152-wide vision selection.
- FlashInfer 0.7 handles full prefill.

The ordinary CLI `--block-size 128` is distinct from these physical sparse-page layouts. Engram remains native and resident in memory; weights are unchanged.

Validation covers six upstream fused-kernel cases, metadata/top-k, 24 eager-versus-graph cases with changing queries, and actual 64-state prefill against an independent FP32 reference using the upstream DeepSeek tolerance (absolute and relative 0.05). See [validation and limits](../docs/validation.md).
