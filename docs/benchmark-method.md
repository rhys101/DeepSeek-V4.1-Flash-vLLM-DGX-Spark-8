# Benchmark method

`bench/v41bench.py`, `bench/prompts-v1.json` and `bench/bench_report.py` are unchanged from [Tony’s pinned repository](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/ca662ac35193c69ace9cee37f13a94abf2eff0fc). Script hashes are recorded in `versions.lock.json`.

The C1–C6 run contains nine workload categories at six concurrency levels: 54 measured batches / 189 requests, preceded by three warmups. The headline is the unweighted mean of code, JSON, math, reasoning, tables, summary, prose and narrative. Counting is reported separately as a ceiling. Each request includes the reference’s deterministic category/concurrency/stream tag. Temperature is zero and thinking is off; token counts come from stream usage records.

Per-stream decode is `(completion_tokens - 1) / (total_seconds - first_token_seconds)`. Aggregate throughput is completion tokens across a batch divided by batch wall time, including prefill. Multiplying average decode by concurrency does not reproduce aggregate throughput. TTFT is time to first token.

The category inputs are 30–303 tokens. All 189 input-token counts match Tony’s boot 10 reference; 86 output lengths differ. There is one measured batch per category/concurrency with no repeated-run confidence interval. The configured 300K context cap does not make these 300K requests.

The same run includes the four original cold-prefill targets, with the unchanged generator, seeds, tags and one-token reply. Actual input lengths are 2,950, 11,592, 46,810 and 93,335 tokens, all matching the reference. Reported prefill rate is actual input tokens / TTFT, an end-to-end measurement rather than isolated kernel throughput. No full 300K request or long-context answer-quality evaluation was performed.

## Reproduce C1–C6 and cold prefill

Run from a quiet client against the documented serving profile, using a fresh output directory and a fresh server process for cold-cache measurements. The benchmark uses deterministic prefixes: changing `--label` or the output directory does not make repeated prompts cold. The recorded main run began after a server restart; its smoke prompts differed from benchmark inputs. C8 used new concurrency-specific prefixes, while its excluded C1 warmups could reuse earlier warmup prefixes.

Run:

```bash
bash bench/run-community.sh http://HEAD:8000/v1 results/my-unique-run spark8-matched
```

Save the resolved server configuration, model/source/image identity, clock observations and health evidence with the result. Vision remains enabled, but the benchmark prompts are text only; image validation is separate.

Recreate the recorded tables with the unchanged upstream formatter:

```bash
python3 bench/bench_report.py results/2026-09-11/bench-spark8.json \
  --vs results/reference/bench-tony-boot10.json
```

## Reproduce C8

The C8 extension uses the same server and source: nine measured batches / 72 requests after three C1 warmups. It does not repeat the prefill sweep.

```bash
python3 -u bench/v41bench.py --base http://HEAD:8000/v1 --model deepseek-v41-flash \
  --label spark8-c8 --out results/my-unique-c8-run --levels 8 --prefill ''
```

Request tags differ with concurrency. C8 is a separate measurement, not a repeated-run confidence interval. Tony’s pinned results stop at C6, so no four-Spark C8 comparison is available.

[Results and comparisons](comparison.md) · [Raw C1–C6/prefill](../results/2026-09-11/) · [Raw C8](../results/2026-09-11/bench-spark8-c8.json)
