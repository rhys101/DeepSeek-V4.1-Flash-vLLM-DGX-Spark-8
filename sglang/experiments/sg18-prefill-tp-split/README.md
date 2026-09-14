# SG18 native prefill TP split — promoted source

**Promoted after successful timing, README and capacity validation.** The
unchanged confirmation showed 21.40–28.05% lower geometric-mean prefill time
on the three primary long-prompt cells against its two controls. The repeat
after a Spark1 reboot passed eight simultaneous 997,100-token contexts, all
cold and cached retrievals, and the requested 1 GiB memory guard. The lowest
sampled available memory was 3.555 GiB. The exact candidate is running and
idle on all eight nodes, with the previous baselines preserved for rollback.
[Current measurements and scope](../../docs/sg18-prefill-results.md) ·
[Promotion record](../../results/sg18-prefill-tp-split/promotion.json).

The snapshot contains the exact 18 production overlays used by the candidate.
Fifteen match the existing [SG18 snapshot](../sg18-indexer-mhc-wo/) byte for byte.
Three differ: scratch initialization in `flash_mla_sm120.py`, the prefill branch
in `deepseek_v4_backend.py`, and the added `spark_prefill_dense.py` helper.
All 241 B12x files and the configuration renderer are inherited unchanged.
Hashes and the two separate patches are recorded in [source-manifest.json](source-manifest.json).
Existing source notices and [SGLang's Apache-2.0 terms](../../LICENSE.sglang) apply.

The scratch fix follows [SGLang PR #39288](https://github.com/sgl-project/sglang/pull/39288).
It initializes persistent split scratch to zero on allocation or growth while
retaining separate SWA/extra-cache buffers and touched-page copying. The fixed
baseline and native candidate both use it. Controlled poisoned-allocation tests
passed 15 cases on each of eight SM121 GPUs; these fixtures demonstrate the
failure mechanism rather than its frequency in ordinary serving.

The prefill branch partitions query rows for the traced native SM120 FP4 MQA
scorer. Quantization, score arithmetic and ragged top-k remain the existing
implementations. The TP collective gathers selected indices and compact block
IDs, from which each rank reconstructs candidate masks. It performs no floating
point reduction. Decode, speculative verification, graph capture, unsupported
layouts, short contexts and small row batches use the original branch.

Qualification passed 45 native GPU cases per rank, including mixed and empty
requests, candidate sources and consumers, plain indexers, both compression
ratios and 997K context. Ordinary and underfilled fixtures require exact raw/page
indices and masks. Six deliberately oversubscribed all-zero tie cases per rank
use an optimal-score oracle because the original top-k also changes indices on
repetition. Those tied indices are not claimed to match bitwise. Serving traces
prove native split/gather dispatch, and 192 C1/C8 decode-dispatch checks passed.
These checks do not establish broad model-quality parity.

Merge [environment.json](environment.json) into the inherited renderer's output
on every rank. The measured settings use TP8/EP4, resident Engram, RoCEnante,
2,048-token prefill chunks, static-five DSpark, 128 request slots, a 1M context
limit and an 8M logical KV pool. The native gates remain 32,768 compressed
context positions and 1,024 rows. The successful capacity repeat used a one-GiB
OS-available guard on every node; retain it for reproduction and preserve the prior deployment
and expert caches for rollback.
This is a source snapshot of the measured integration; a generalized installer
has not been validated.

Verify package identity, reconstruct both patches and check partition properties
without importing Torch or touching a GPU:

```bash
python3 sglang/experiments/sg18-prefill-tp-split/verify-source.py
python3 sglang/experiments/sg18-prefill-tp-split/verify-results.py
python3 sglang/experiments/sg18-prefill-tp-split/verify-promotion.py
```

The [results package](../../results/sg18-prefill-tp-split/) preserves all cold
blocks, README repetitions, frozen prompts, clients, raw guard samples and
component receipts, including the initial failures and interrupted timing attempt.
The offline verifier replays the original accounting and comparison scripts in
a temporary directory. It does not repeat GPU execution or run benchmark clients.

The confirmation used the same frozen cold prompts, warmups, three trials per
cell and pinned README clients. The initial cold comparison retained a C8/32K
regression against its preceding control and inconsistent baseline timing; it
is not discarded or averaged into the confirmation. The first capacity attempt
is retained as a failure: eight distinct 997,100-token prompts were submitted,
but zero retrievals completed and no simultaneous residency was proved. The
OS-only records identify the head-node threshold crossing, without identifying
the allocation responsible or isolating the split as its cause.

The subsequent repeat used the exact same eight prompt bytes and unchanged
client and candidate source after the user rebooted Spark1. It completed
8/8 cold retrievals and 8/8 cached 1,024-token responses. Thirty fresh load
samples proved all eight contexts active together. All 1,299 guard samples
passed; they also stayed above the earlier two-GiB floor. This does not prove
the cause of the first attempt's memory drop. The historical evaluation and
its baseline-restored decision remain immutable snapshots of that first
attempt. The promotion record links the successful repeat and final audit.
