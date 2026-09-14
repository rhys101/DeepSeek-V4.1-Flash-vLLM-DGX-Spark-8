# TP8 prefill evaluation — 14 September 2026

**Completed: the capacity memory gate failed; the prefill change is not promoted.
The qualified scratch-fixed SG18 baseline is restored and verified on all eight nodes.**

Cold and README confirmation passed both controls, with 21.40–28.05% lower
geometric-mean time on the primary long-prompt workloads. However, the final
eight-request near-1M test crossed the fixed 2 GiB OS-available threshold on the
head node before any retrieval completed. The guard stopped all eight candidate
containers. The timing gains remain valid measurements; they do not clear the
deployment gate. The public repository has not been changed for this candidate.

The native FP4 indexer split passed its component and serving-dispatch checks.
The first complete cold-prefill block reduced the geometric mean of the three
long-workload times by 11.24% against the preceding corrected baseline and
22.79% against the following control. The 32K C8 batch was 5.08% slower against
the preceding control and 1.83% slower against the following control. The
initial comparison did not clear the short-workload gate. The candidate's unchanged README
suite passed comparison against both controls, with no new answer variants.
The one planned cold confirmation, with unchanged source and thresholds, now
passes both controls. Each arm used a fresh serving process followed directly
by the same cold-suite warmups and trials, then its full README suite. The
README comparison also passes both controls, with no new answer variants.

Both arms include the independently validated scratch-initialization fix for
[SGLang PR #39288](https://github.com/sgl-project/sglang/pull/39288). All other deployment settings remain fixed: eight Sparks,
TP8/EP4, resident Engram, RoCEnante, static five-token DSpark, 2,048-token
prefill chunks, 128 request slots, 1M context and an 8M logical token pool.

| Cold input workload | Baseline before, seconds | Native split, seconds | Baseline after, seconds | Time change vs before | Time change vs after |
|---|---:|---:|---:|---:|---:|
| C1 · 4,096 tokens | 0.97968 | 0.92242 | 0.93883 | −5.85% | −1.75% |
| C1 · 32,768 tokens | 7.89530 | 7.99859 | 8.28808 | +1.31% | −3.49% |
| C1 · 131,072 tokens | 38.21729 | 34.45486 | 42.67493 | −9.84% | −19.26% |
| C1 · 299,008 tokens | 115.88941 | 99.52117 | 149.11441 | −14.12% | −33.26% |
| C8 · 32,768 tokens each | 60.56303 | 63.64257 | 62.50026 | +5.08% | +1.83% |
| C8 · 131,072 tokens each | 307.64131 | 277.86141 | 325.34942 | −9.68% | −14.60% |

Each cell contains three measured trials after one excluded warmup. The
primary time ends when the last request reports its first generated token;
each request generates one token. Inputs, client source, trial ordering and
cache-flush procedure are identical. Every reported cached-token count was
zero in all three blocks. The following control passed independent validation
and retained at least 5.05 GiB of OS-available memory per node. Its C1/299K
trials were 177.96, 138.19 and 131.19 seconds; all remain included. The before
and after controls are not averaged together. A separate timestamp check
confirms they temporally bracket the candidate. These synthetic timings are separate from README decode
rates and the README retrieval timings.

The long-workload input rates calculated from those mean times are approximately
3,804 tokens/s at C1/131K, 3,004 tokens/s at C1/299K and 3,774 aggregate tokens/s
at C8/131K. These are initial comparisons, not accepted deployment claims.

The correctness record includes 16,404 CPU partition cases and 45 GPU cases on
each of eight ranks. Each rank passed 39 exact raw/page-index comparisons and
six deliberately oversubscribed all-zero tie cases using the documented
optimal-selection oracle. Valid scores and candidate masks remained exact.
The original top-k routine itself was not repeatable on those fully tied
fixtures; exact tied-index identity is therefore not claimed. Cases include
997K context, mixed and empty requests, both compression ratios, and source,
consumer and plain indexers.

Serving traces prove the intended row split and gathers on all eight ranks.
The separate decode check passed all 192 dispatch comparisons. Startup checks
passed all 500 serving settings and verified source and expert-cache hashes.
The candidate cold run retained at least 8.47 GiB of OS-available memory per
node; its README suite retained at least 7.72 GiB. The stop threshold is 2 GiB.

The README suite passed text, image, structured-JSON and tool-round-trip checks
before and after long-context retrieval, plus 128/128 simultaneous arithmetic
requests. All retrieval answers at 32K, 131K and 299K matched. The before-and-after
baseline answer comparison found no new variants across 47 checks. C1/C8
coding and prose had no decline over 3% in both benchmark repetitions against
either control. The following README control passed independently and retained
at least 5.16 GiB of OS-available memory per node.
These are bounded correctness checks, not a broad reasoning-quality evaluation.

Separate C8/32K traces found no split in the interior and one packed source
gather per rank in the 20-step boundary capture. The largest collective kernel
was 1.64 milliseconds. That observation alone does not explain a roughly
three-second batch difference. The traces did not support the proposed mixed
request padding explanation. The 32K threshold and candidate source remain
unchanged.

All failures and discarded hypotheses remain in the record. The first
prototype targeted an inactive Torch fallback and was never deployed. An
interrupted timing attempt retained six warmups and seven measured batches;
its results are not mixed into the complete block above. It ended with a
client `BrokenPipeError` around the Mac's offline period while workers remained
healthy. Subsequent jobs use detached remote controllers and durable logs.
One diagnostic capture failed on an idle-state trigger; its corrected capture
passed.

The initial runs had different preceding workload histories: their recorded
prefill counters before timing were 487,323 tokens for the preceding baseline,
4,591,549 for the candidate, and zero for the following baseline. Their role in
the timing drift has not been established. The confirmation aligns the
preparation procedure and checks for zero recorded prefill before timing; it
does not claim identical OS-cache, allocator or thermal state. The completed
following baseline is the preceding control for this confirmation. A further
fresh baseline has now completed after the candidate, so the new cold comparisons
also have controls on both sides. All original measurements remain in the record.

The confirmation candidate completed all 24 cold batches and passed independent
SSE accounting and raw guard validation. It started with zero recorded prefill
on a fresh serving process and used the identical client, inputs, warmups and
three-trial protocol. All reported cached-token counts were zero. Minimum
OS-available memory was 8.94 GiB per node across 538 guard samples; worker process
identities stayed unchanged, with no recorded OOM or restart.

| Confirmation cold workload | Fixed before, seconds | Native confirmation, seconds | Fixed after, seconds | Time change vs before | Time change vs after |
|---|---:|---:|---:|---:|---:|
| C1 · 4,096 tokens | 0.93883 | 0.92493 | 1.06039 | −1.48% | −12.77% |
| C1 · 32,768 tokens | 8.28808 | 7.80877 | 8.13018 | −5.78% | −3.95% |
| C1 · 131,072 tokens | 42.67493 | 33.54887 | 39.27536 | −21.39% | −14.58% |
| C1 · 299,008 tokens | 149.11441 | 86.46150 | 128.45972 | −42.02% | −32.69% |
| C8 · 32,768 tokens each | 62.50026 | 61.87907 | 62.30854 | −0.99% | −0.69% |
| C8 · 131,072 tokens each | 325.34942 | 265.86832 | 314.72990 | −18.28% | −15.52% |

The completed cold confirmation passes both controls: the long-workload geometric
mean is 28.05% and 21.40% lower, respectively, and no short-cell mean regresses.
The timestamps independently confirm that the controls bracket the candidate.
The preceding 299K control retains its slow first trial. The following control's
299K trials were 133.71, 125.43 and 126.24 seconds; its 4K trials were 1.26084,
0.96858 and 0.95175 seconds. All remain included. The following run passed
independent accounting with zero cached tokens and a 5.26 GiB memory minimum
across 645 guard samples, with unchanged worker identities and no recorded OOM
or restart. This confirmation clears the cold timing gate; the cause of the
earlier drift remains unproven. The original candidate and controls remain
separate from these measurements.

The README confirmation passed independent validation of the unchanged 35-file
reference kit, both capability passes, all 128 arithmetic requests and the three
long-context retrievals. Its minimum OS-available memory was 8.51 GiB per node.
C1 coding decode averaged 131.33 and 131.72 tokens/s in the two repetitions;
C8 full-batch coding aggregate averaged 498.06 and 508.79 tokens/s. Comparison
against both confirmation controls passed, with no repeatable C1/C8 coding or
prose decline over 3%. All 47 answer comparisons match observed control variants.
The following README control passed both capability suites, all 128 arithmetic
requests and all three retrieval checks, with at least 6.29 GiB OS-available
memory across 106 guard samples. Its initial and repeated C1 coding means were
128.19 and 128.40 tokens/s; C8 coding aggregate means were 474.80 and 500.67
tokens/s. Every slower trial is retained. The recorded timestamps confirm the
README controls bracket the candidate.

An interim audit after the candidate finished found all 261 staged files and
expert caches unchanged, healthy transport markers, and no remaining experiment
jobs. The original and fixed baseline containers and caches remained intact.
Fresh-start logs confirmed split dispatch on all eight ranks with the original
32K/1,024-row gates and draft fallback. The candidate is now stopped and preserved
after the fixed baseline completed the following README control. Its startup passed the
source/cache and serving-setting checks; both preflight and launch confirmed
zero prior prefill before the unchanged cold suite began. This interim audit
does not replace the final handoff audit after the remaining work.

The completed confirmation cleared the prerequisite gates for the eight
simultaneous near-1M capacity test. The candidate restart passed all 500 serving
settings and source/cache checks on eight ranks, with a 6.77 GiB startup memory
minimum. The durable capacity job started at 20:31 UTC and submitted eight distinct
997,100-token prompts, totaling 7,976,800 input tokens.

At 20:40:05 UTC the guard observed 1.95256 GiB OS-available memory on rank 0,
below the unchanged 2 GiB threshold. Its 142 samples show unchanged, running
worker processes with no recorded OOM kill or restart before the stop. The other
seven ranks had 9.39–12.15 GiB available at the failing sample. No cold retrieval
completed, the cached phase never started, and none of the 472 load samples
proved eight simultaneous resident contexts. The last successful load response
still showed seven waiting requests and 983,040 used tokens; its approximately
7.96M active-token counter alone is not residency proof.

The guard stopped all eight containers; each exited with code 137 after its
shutdown deadline, with Docker's OOM-kill flag false. The resulting client
disconnects are retained. Sequential shutdown exceeded the controller's 90-second
finalizer wait, so the raw guard files were recovered directly from the head.
All client records, frozen prompts, guard samples and eight worker logs are
preserved. An independent reconstruction confirms the memory-gate failure and
finds no CUDA out-of-memory, illegal-access or scheduler-exception markers in
the collected worker logs. Distributed teardown warnings remain in the logs.

The OS-only samples do not identify the allocation responsible for the head-node
pressure. This experiment did not run the matched fixed baseline on the same
near-1M capacity workload, so it does not isolate the split as the cause. Memory
attribution on the head is the next investigation before another capacity
qualification; lowering the guard or reducing the promised capacity would change
the acceptance criteria.

Baseline restoration started at 20:43 UTC and completed at 20:50 UTC. All 500
serving settings matched, and source/cache checks passed on all eight ranks.
Minimum OS-available memory during restoration was 10.14080 GiB. The final
handoff audit passed at 20:53 UTC with a minimum of 10.13872 GiB: the fixed
baseline was the sole healthy GPU service on every node, requests were idle,
and no experiment jobs remained on the client, head or local machine. All 260
baseline source/configuration files and its two expert caches matched. The
original SG17, SG18 and failed native-candidate containers, source and expert
caches remained stopped and preserved.

The final decision is **NOT_PROMOTED_BASELINE_RESTORED**. No candidate source,
results or headline changes were pushed to the public repository. The source
and evidence bundle are retained as a local review draft, with the initial
regression, interrupted attempt and capacity failure included. The offline
verifier reproduced six complete cold blocks, six README suites, all 13
interrupted batches and the failed capacity evidence using the original saved
validators; original SG18 controls remain separate from corrected controls.
