# TP8 prefill indexer split — execution plan

**Completed: not promoted after the near-1M capacity memory gate failed.
The qualified scratch-fixed SG18 baseline is restored and verified on all eight nodes.**

**14 September 2026 · baseline: SG18 plus the validated scratch initialization fix.** Improve long-prompt prefill
by distributing the indexer's replicated query-row work across the eight Sparks.
Tony's TP4 result was about 6.3% at 131K; the improvement on this implementation
must be measured. The current SG18 containers remain the rollback target.

## Fixed deployment

Use the current image, checkpoint, SG18 indexer/mHC/WO kernels, TP8/EP4 topology,
resident Engram, RoCEnante transport, static five-token DSpark, 2,048-token
prefill chunks, 128 request slots, 1,000,000-token context limit and 8,000,000-token
logical KV pool. Retain the two-GiB OS-available stop threshold on every node.
Archive source, configuration and expert-autotune hashes for both arms.

The takeover audit passed on all eight ranks at 12:40 UTC, with a minimum of
10.14 GiB available per node and no active experiment requests or controllers.

## Implementation

The completed all-rank trace shows the active prefill path is
`_low_ratio_index_topk_dense`, calling DeepGEMM's SM120 FP4 MQA logits kernel.
The first, undeployed prototype targeted the Torch fallback and is withdrawn
from consideration. The indexer has 32 heads, 128 dimensions and top-k 512;
candidate filtering retains up to 2,048 blocks of eight positions.

1. Retarget query-row partitioning to the traced native FP4 path, retaining the
   query/key quantization, score arithmetic, selection and candidate semantics.
   Validate exact results before choosing partition boundaries or gates.
2. Gather selected indices and compact candidate block IDs over the actual TP
   communicator. Reconstruct the original boolean candidate masks exactly.
3. Keep short requests, decode/verification, graph capture and unsupported
   parallel layouts on their existing paths. Agree the enable flag and gate
   settings across ranks during initialization.
4. Prove actual split/gather dispatch in serving traces before timing claims.

A separate upstream audit found missing scratch initialization from SGLang
PR #39288. That narrow fix passed controlled poisoned-allocation tests on all
eight GPUs, the full README suite and decode-dispatch checks. Both prefill arms
inherit the fix in the isolated `scratchfix01` baseline.
The completed old-SG18 controls stay archived and must not be represented as
measurements of the corrected baseline. The audit and initial plan are retained.

## Correctness before serving

Run CPU partition/scatter checks, then tests on idle SM121 GPUs using the actual
DeepGEMM FP4 score and SGLang candidate-selection functions. Cover both compression ratios,
candidate source and consumer layers, newest-block retention, underfilled top-k,
ties, mixed request lengths, chunk boundaries, empty ranks and repeated inputs.

Require exact selected raw indices, page indices and reconstructed candidate
masks against repeated original calls. Test the real eight-rank collective path
with distinct row markers and identical replicated inputs; check every rank's
result. Retain all failures. Any arithmetic or selection mismatch must be fixed
or excluded through a justified fallback before full-server measurements.

The first GPU run exposed a limit in the original repeatability requirement:
the original top-k kernel chooses different indices on repeated, deliberately
all-zero inputs when more than 512 positions tie. A separate eight-rank
diagnostic confirmed exact optimal score multisets, valid unique selections and
correct page mappings for both implementations, with exact ordinary-case
indices and masks. Only these deliberately oversubscribed, fully tied fixtures
use that semantic oracle. All ordinary and underfilled fixtures retain exact
raw/page equality; valid native scores and candidate masks remain exact.
Tied indices are not claimed to be bitwise identical. Full-model answer checks
remain a separate requirement. The initial failure and diagnostic are retained.

## Measurements

Run every serving workload under the all-rank memory/health guard, one workload
at a time, from the separate CPU client. Preserve corrected baseline controls before
and after the candidate to expose drift. Profiling is separate from timed runs.

- **Cold prefill:** frozen, distinct synthetic document prompts at 4K, 32K,
  128K and approximately 299K tokens at C1; 32K and 128K at C8. Flush the prefix
  cache between trials, retain token counts and prefix-cache observations, and
  collect three measured trials per cell after an excluded warmup. Record TTFT,
  complete request time and aggregate input processing rate; output is minimal.
- **Dispatch:** cold-prefill traces on all ranks prove the intended score work
  and gathers ran. A separate decode trace verifies unchanged expert dispatch.
- **README suite:** use the existing pinned Tony coding and sparkDash prose
  clients, including their original warmups, trials, capability checks and
  long-context retrieval. Keep the benchmark source and prompt hashes unchanged.
- **Answers:** compare deterministic retrieval and short-answer fixtures across
  original/candidate runs at short and long contexts. Investigate any changed
  outputs against baseline repetition; component checks alone do not establish
  full-model quality.

The primary performance criterion is at least **3% lower geometric-mean cold
prefill time** across C1/128K, C1/~299K and C8/128K, against both corrected control
blocks. Inspect every cell. A repeatable regression greater than 3% in the short
prefill checks or README C1/C8 coding/prose blocks prevents promotion until
explained and resolved. If drift makes the result ambiguous, perform one
explicitly labelled confirmation block and preserve all previous measurements.
An early screen can reject a clearly slower implementation before the full suite.

## Eight simultaneous near-1M contexts

After the candidate passes the previous gates, submit eight distinct prompts
with approximately 997K input tokens each, keeping room for output, draft tokens
and page rounding. Use the prior capacity test's three-record retrieval pattern,
with distinct early identifiers and independently seeded values.

Require all eight cold retrievals to pass, then use a longer cached output pass
to establish simultaneous residency: eight active requests and more than 7.9M
used tokens in the server's load records, rather than merely eight submitted
requests. Check counts, records, completion and memory throughout; retain raw
streams and server evidence. This is a capacity/retrieval test on synthetic
documents, not a broad million-token reasoning evaluation. It may take hours.

## Decision and handoff

Promote only after correctness, measured performance, README checks, capacity and
final all-rank health/source/cache audits pass. Preserve the previous SG18
containers and caches for rollback. Publish the narrow source change and full
results, including slower trials and limitations; update headline claims only
where the new measurements support them.

If the candidate fails a gate or offers no useful gain, restore SG18 with the
validated scratch fix (`scratchfix01`), verify it
is the sole healthy GPU service on all eight nodes, and report what failed and
what was learned. No source or cache files in the preserved baseline are edited.

## Execution record

- Takeover audit: passed.
- Source inspection: exact live indexer and collective files captured.
- Original SG18 README and cold-prefill controls: completed and archived.
- Trace correction: all eight ranks use the native FP4 scorer. The first Torch
  fallback prototype, `prefill01`, was never staged or run and is unqualified.
- Scratch initialization fix: passed 15 controlled GPU cases per rank, the full
  README suite (including C128 and 32K/131K/299K retrieval), and 192 decode-dispatch checks.
- Corrected cold-prefill baseline: passed all 18 measured cells and six excluded
  warmups using the same frozen prompts and client, with zero cached input tokens.
  Mean times: 38.22 seconds at C1/131K, 115.89 seconds at C1/299K and 307.64 seconds
  at C8/131K. Minimum OS-available memory was 3.59 GiB per node. The 299K trials
  ranged from 112.38 to 122.86 seconds; all remain included.
- Native candidate: staged as `prefillnative01`; two files changed from
  `scratchfix01`. Full component qualification and serving startup passed.
- Native partition checks: 16,404 CPU cases passed; patch reconstruction and
  source-scope checks passed.
- Native GPU qualification: passed 45 cases per rank, including 997K context,
  mixed/empty requests, uneven partitions, both compression ratios, candidate
  sources, consumers and plain indexers, and fallback dispatch. Each rank passed 39 direct exact
  raw/page comparisons and six checks using the documented all-zero tie oracle.
  Valid scores and candidate masks remained exact. All gathered candidate results
  matched across ranks. Minimum OS-available memory was 76.25 GiB per node with
  the model unloaded.
- Native serving trace: passed on all eight ranks. Two late C1-prefill steps
  contain eight full-chunk gathers (two candidate-source and six plain-index
  gathers), each collecting 256 local rows into 2,048 global rows. Eight
  128-row late-layer tails remain on the original path, as required by the gate.
  The earlier expectation of 16 gathers was corrected after full trace inspection.
- Candidate startup and restart: passed all 500 settings and source/cache checks,
  with identical source after the additional plain-mode GPU qualification.
- Fresh decode trace: passed all 192 dispatch checks across C1/C8 and eight ranks.
- First candidate timing block: interrupted after six warmups and seven measured
  batches by a client `BrokenPipeError`. Every completed batch is preserved;
  independent SSE accounting passed, with zero cached tokens and at least
  8.42 GiB OS-available memory. The workers remained healthy and idle afterward.
  Stale local SSH progress was initially mistaken for a possible model hang.
  The exact failing stack frame was not retained; a broken reporting connection
  is the likely cause.
- Candidate timing restart: completed as `prefill-native-v02`, with the same client,
  prompts and trial order. A detached controller on the benchmark host writes
  progress to files there and closes its guard on completion. The interrupted
  results remain separate and are not combined with the new block's means.
- Initial candidate screen: **not cleared**. All 24 batches passed independent
  SSE accounting, with zero cached input tokens and a minimum of 8.47 GiB
  OS-available memory. Mean times were 34.45 seconds at C1/131K, 99.52 seconds at
  C1/299K and 277.86 seconds at C8/131K: a geometric-mean time reduction of 11.24%
  against the preceding corrected baseline. C8/32K increased from 60.56 to
  63.64 seconds, a repeatable 5.08% regression. All trials remain included.
  The full comparison with the following baseline is recorded below.
- C8/32K diagnostic traces: completed on all eight ranks. Interior steps never
  activated the split. A 20-step boundary capture shows one packed source gather
  per rank, collecting 256 rows into 2,048. The largest collective GPU kernel
  was 1.64 milliseconds; this observation alone does not explain the roughly
  three-second batch difference. The trace did not support the proposed mixed
  request padding explanation. Source and the 32K threshold remain unchanged.
  One earlier boundary capture failed because its trigger sampled an idle state;
  that failure is retained, and the corrected capture passed.
- Candidate README suite: passed independent validation with the unchanged
  pinned client and prompts. Both capability passes, C128 arithmetic (128/128)
  and all three retrieval cases passed. Minimum OS-available memory was 7.72 GiB.
  Neither C1 nor C8 coding/prose declined by more than 3% in both benchmark
  repetitions against the preceding corrected baseline. The completed comparison
  with the following baseline and short-answer review is recorded below.
- Following cold control: passed all 24 batches and independent accounting,
  with zero cached tokens and a minimum of 5.05 GiB OS-available memory. Mean
  long times were 42.67 seconds at C1/131K, 149.11 seconds at C1/299K and 325.35
  seconds at C8/131K. The 299K trials were 177.96, 138.19 and 131.19 seconds;
  all remain included. Source, expert-cache and all 500 settings checks passed
  after restarting the preserved baseline. The following README suite passed
  independently, with a minimum of 5.16 GiB OS-available memory.
- Initial full cold comparison: **did not clear the short-workload gate**. The candidate reduced the
  long-workload geometric mean by 11.24% and 22.79% against the preceding and
  following controls, respectively. Its C8/32K time increased by 5.08% and 1.83%,
  respectively. A separate timestamp check confirms the controls temporally
  bracket the candidate. The controls are not averaged together.
- Full README comparison: passed against both controls, with zero new answer
  variants and no repeatable C1/C8 decline over 3%. The recorded timestamps also
  confirm that the README controls bracket the candidate.
- Confirmation: the one planned series completed with unchanged source and
  thresholds. The completed following baseline becomes its preceding control;
  a further fresh baseline followed the candidate. Each arm started a fresh
  serving process and runs the same cold-suite warmups directly, then its full
  README suite. The earlier blocks had different preparation workloads; their
  role in drift is unproven. All original measurements remain in the record.
- Confirmation candidate cold block: completed all 24 batches and passed
  independent accounting, with zero cached input tokens and a minimum of
  8.94 GiB OS-available memory. Mean times were 33.55 seconds at C1/131K,
  86.46 at C1/299K and 265.87 at C8/131K. C8/32K averaged 61.88 seconds.
  The preceding fresh-control screen passes all timing gates; the following
  control subsequently completed and passed the full comparison below. The unchanged full README suite completed and
  passed independent validation, with at least 8.51 GiB OS-available memory.
- Confirmation candidate interim audit: passed all 261 source files and expert
  caches, preserved baseline source/cache identities, transport/health checks
  and idle-process checks. Fresh-start logs confirm the split ran on every rank.
  The native candidate is stopped and preserved. The fixed baseline's fresh
  startup passed source/cache and all 500 serving-setting checks, then both
  preflight and launch confirmed zero prior prefill. The following cold control
  completed with the unchanged protocol. The final handoff audit remains required.
- Confirmation timing decision: **passed against both controls**. Long-workload
  geometric-mean time fell by 28.05% against the preceding control and 21.40%
  against the following control. Every short-cell gate passed; C8/32K was 0.99%
  and 0.69% faster, respectively. The following control passed all 24 batches
  and independent accounting, with zero cached tokens, unchanged worker identities
  and at least 5.26 GiB OS-available memory across 645 guard samples. Both controls
  temporally bracket the candidate. All initial results and slower trials remain
  retained; the cause of the original drift remains unproven.
- Confirmation README decision: **passed against both controls**. Both repetitions
  have no repeatable C1/C8 coding or prose regression over 3%; all 47 answer
  comparisons match observed control variants. The following baseline's unchanged
  suite passed both capability checks, 128/128 arithmetic and all three retrievals,
  with at least 6.29 GiB available per node. Its controls also temporally bracket
  the candidate. The capacity prerequisite record passed 45 evidence-file checks;
  the fixed baseline was stopped and preserved for a qualified candidate restart.
- Local source review: prepared an isolated draft containing the exact 18
  production overlays, separate scratch and prefill patches, inherited B12x
  hashes and the unchanged configuration renderer. Patch reconstruction,
  source identity, unchanged fallback checks and 16,404 partition cases passed;
  all 11 repository configuration tests passed. The draft is not published.
- Capacity: **failed the memory gate**. Prerequisite gates and candidate restart passed, including all 500
  settings and eight-rank source/cache checks, with at least 6.77 GiB available
  during startup. The durable test launched at 20:31 UTC with eight distinct
  997,100-token prompts. At 20:40:05 UTC rank 0 fell to 1.95256 GiB OS-available,
  below the unchanged 2 GiB threshold; the other seven ranks remained above
  9.39 GiB. The guard stopped all eight containers. No retrieval completed and
  no simultaneous residency was proved. Independent reconstruction of all 142
  guard samples, 472 load samples and worker logs passed; no OOM kill was
  recorded. The head-node allocation responsible remains unknown. The failed
  client and controller shutdown-timeout records remain preserved.
- Handoff: **passed**. The candidate is not promoted. Restoration of the qualified
  `scratchfix01` baseline started at 20:43 UTC and completed at 20:50 UTC.
  Source/cache and all 500 settings checks passed. The final audit at 20:53 UTC
  found the baseline healthy and idle on all eight nodes, with at least 10.14 GiB
  OS-available and no experiment jobs remaining. Baseline and candidate source
  and cache identities remain preserved; all other GPU containers are stopped.
  The public repository has not been changed for this candidate. The local
  source/evidence draft retains all measurements, including the failed capacity run.
