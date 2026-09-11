# Eight million token results: DeepSeek V4.1 Flash on eight DGX Sparks

**Eight distinct 997,097-token prompts passed the capacity and retrieval test: 7,976,776 input tokens in an 8M-token cache pool.** All eight requests were observed running together. The lowest sampled OS-available memory was **7.83 GiB on one Spark**; the other seven Sparks' minima were **10.70–11.09 GiB each**.

Measured 11 September 2026 UTC with SGLang TP8/EP4, native resident Engram and DSpark five-token drafting. This is the separate SG8 capacity experiment using the [SG5 EP4 source composition](build-and-pins.md). It extends the measured context capacity beyond the default deployment profile.

## Results

All throughput figures below are aggregate across the eight-Spark cluster unless explicitly labelled otherwise.

| Measurement | Result |
|---|---:|
| Distinct input tokens per request | 997,097 |
| Eight-request input total | **7,976,776** |
| Context limit / shared logical pool | 1,000,000 per request / 8,000,000 total |
| Logged prefill peak within the first prompt's initial 128K | **5,076.03 tokens/s** |
| Cold input rate over the complete eight-request run | **1,513.50 tokens/s** |
| Cold time until every request had first output | **87.84 minutes** |
| All cold responses complete | 87.89 minutes |
| Longer cached output, including request startup | **138.31 tokens/s** |
| One server decode interval with eight requests active | **158.12 tokens/s** |
| That C8 interval divided by eight active requests | 19.77 tokens/s per request |
| Lowest sampled free OS memory on any Spark | **7.83 GiB per Spark** |

[Machine-readable results](../results/eight-million/summary.json) · [Server excerpts](../results/eight-million/server-excerpts.log) · [Concurrent residency evidence](../results/eight-million/residency.json).

## Prefill progression

The **5,076.03 tokens/s peak occurred near 4,096 processed tokens**, within the first 128,000 tokens of the first prompt. It is a server-reported chunk rate, not the average over all 128K tokens. The median logged chunk rate in that initial range was **4,278.57 tokens/s**.

| Approximate position within the first prompt | Median logged chunk rate (tokens/s) |
|---|---:|
| 0–128K | 4,278.57 |
| 128–256K | 2,246.63 |
| 256–512K | 1,643.65 |
| 512–768K | 1,122.36 |
| 768K–1M | 823.40 |

[First-prompt chunk measurements](../results/eight-million/first-prompt-prefill.csv). Positions use cumulative logged new-token counts; final-chunk padding or mixed scheduling means those counts are approximate. API usage supplies the exact 997,097-token prompt length.

The overall cold rate divides 7,976,776 input tokens by 5,270.43 seconds, measured from the earliest request submission until the last request's first nonempty output. It includes transfer, tokenization, scheduling and any interleaved decoding. Prefill proceeded predominantly one long prompt at a time. Eight submitted requests did not provide eight independent prefill workers, and earlier output streams paused while later prompts loaded.

## Cached output speed

The longer cached pass requested **1,024 output tokens from each of eight requests**. It delivered **8,192 output tokens in 59.23 seconds**, or **138.31 tokens/s including startup**. First-output latency was 16.96–17.96 seconds.

A server log interval at 23:21:23 UTC recorded **eight active requests, 7,980,032 used tokens and 158.12 generated tokens/s**. Dividing by eight gives 19.77 tokens/s per active request. This is one observed interval; its throughput window may include the end of request admission. It is not a many-run steady-state average or an independently measured rate for every stream. The load recorder captured 24 qualifying samples covering five distinct server snapshots with eight active requests and more than 7.9M used tokens.

The initial 256-token cached pass delivered 2,048 tokens in 25.60 seconds: **79.99 tokens/s including startup**. Buffered output bursts inflated its apparent rate after first output to about 230 tokens/s. That burst-timing figure is excluded from decode-speed claims. The shorter replay also allowed earlier requests to finish before all eight were admitted; the longer pass supplied the C8 evidence.

This synthetic retrieval/forced-output workload differs from the repository's prose and coding benchmarks. After the initial correct JSON response, generation was forced past the normal stopping point to extend the decode measurement.

## Adding 50 tokens to cached contexts: C1, then C8

These tests used the same eight-Spark TP8/EP4 setup and 8M-token pool. One request ran first, followed by eight requests submitted together. API usage confirmed exactly **997,147 input tokens per request**, an increase of 50.

| Load | Time to first output | Complete 20-token response | Retrieval |
|---|---:|---:|---:|
| C1 | **2.44 s** | **2.88 s** | 1/1 correct |
| C8 | **13.51–14.26 s** | **All eight in 14.64 s** | 8/8 correct |

Every request reused **996,864 cached tokens** and recomputed **283 tokens**: the new 50 tokens plus the prior uncached tail. These are client-visible API timings, including transfer, tokenization, scheduling, tail recomputation and response delivery, rather than timing only 50 tokens of GPU work.

The C1 suffix contained fifty ` the` tokens and the C8 suffix contained fifty ` and` tokens, preventing the C1 extension from pre-caching the C8 extension. Both lengths were verified by usage counts. These short responses measure completion of eight submitted requests; they do not imply eight requests remained in decode throughout the interval.

## Memory per Spark

The measurements are **OS `MemAvailable` on each individual Spark**, in GiB. **7.83 GiB is the lowest per-node sample, not a cluster-wide total.** It is also not the KV-cache allocation size.

| Spark | Rank | Minimum sampled available memory (GiB) |
|---|---:|---:|
| Spark 1 | 0 | 7.831 |
| Spark 2 | 1 | 10.700 |
| Spark 3 | 2 | 10.783 |
| Spark 4 | 3 | 11.093 |
| Spark 5 | 4 | 10.953 |
| Spark 6 | 5 | 10.995 |
| Spark 7 | 6 | 10.815 |
| Spark 8 | 7 | 10.779 |

No node ran out of memory or restarted, and all eight containers were healthy after the tests. The later output and append tests had a minimum of **22.238 GiB per node**. The capacity audit combines one-second OS samples with all guard generations; it cannot rule out lower values between samples.

The operator lowered the emergency reserve threshold from 8 to 4 to **2 GiB per node** during the capacity experiment. This was a stop threshold, not the observed free-memory minimum. All monitors completed normally after the tests.

## Tested configuration

| Setting | Capacity experiment |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4.1-Flash`, revision `df42c109f1defefcbfcedbe7d905718a12266e40` |
| Hardware / parallelism | Eight GB10/SM121 DGX Sparks; TP8, EP4, MoE-TP2 |
| Engram | Native resident owned rows on each Spark |
| Context limit | 1,000,000 tokens per request, including output |
| Shared logical token pool | 8,000,000 tokens |
| Concurrent request slots | 8 |
| Prefill chunk / maximum prefill tokens | 2,048 / 2,048 |
| Static memory fraction | 0.80 |
| Speculation | DSpark, five-token drafting |
| Final emergency OS reserve | 2 GiB per node |

The model precision and production source overlays were retained from SG5: checkpoint MXFP4 experts, FP8 dense weights, BF16 activation dtype, automatic KV selection resolved to FP8 E4M3, and the existing expert computation path. [Precision and validation details](validation.md).

The experiment used image ID `sha256:e7681b5276525821f9be286ed3bf0f51acb5239da27f69f60fdeb57e68c05581` with the six recorded production source overlays. Their original/final hashes are in the [result summary](../results/eight-million/summary.json); an image ID alone does not identify this mounted source composition. [Build and pin history](build-and-pins.md).

The public launcher still validates the standard **300K-context / 3.2M-pool / 8,192-token-prefill / 13-GiB-reserve profile**. This report records the separately validated capacity experiment; editing only the example JSON will not make the current launcher accept these larger settings.

## Workload and limits

The cold requests used distinct early identifiers, 997,000 repeated ` the` filler units, and three independently generated eight-digit records near the beginning, middle and end. For each request, one Python `random.Random(8161900 + request_index)` instance generated the three values with consecutive `randrange(10000000, 100000000)` calls. Requests used temperature zero, thinking off and streamed responses. Exact API prompt lengths were 997,097 tokens; the cold output budget was 64 and every response used 20 output tokens. The cached passes reused each request's own prompt. Using eight identical prompts would have permitted shared-prefix reuse and would not establish the same distinct-context capacity.

All three records matched exactly in every request of the cold, 256-token replay, 1,024-token replay, C1 append and C8 append phases. Cold API snapshots showed eight running requests and **7,976,960 used tokens**; the longer output pass independently confirmed eight active contexts. Prompt lengths left room in the configured limits for output, draft slots and allocator rounding.

These are capacity, basic retrieval and timing measurements on repeated filler. They do not establish broad reasoning or coding quality over arbitrary million-token documents. Results are single runs without confidence intervals.

## Evidence

- [Summary, per-request counts, retrieval values, timings and source hashes](../results/eight-million/summary.json)
- [First-prompt prefill samples](../results/eight-million/first-prompt-prefill.csv)
- [Cold and longer-output residency receipts](../results/eight-million/residency.json)
- [Relevant prefill and output server log lines](../results/eight-million/server-excerpts.log)
- [Hashes of preserved source artifacts and published extracts](../results/eight-million/provenance.json)
