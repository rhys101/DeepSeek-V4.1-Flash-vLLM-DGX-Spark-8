# Speed versus model capability

**Faster output is useful only if the model remains capable enough for the intended task. A disproportionate loss of answer quality, reasoning, tool reliability or context capability outweighs a throughput gain.** Performance and quality must be reported separately; a higher token rate is not evidence of a better model.

## Current optimizations

The deployed NCCL settings change communication buffers, protocol selection and channel count. The small-M b12x route changes the dense MXFP8 kernel while retaining the existing weight and activation quantization formats. Neither change deliberately reduces model precision, expert count, context allowance or verification strength. Different kernels and reduction orders can still change floating-point results; these changes are not a claim of bitwise equivalence or proven quality parity.

The current checks cover independent kernel references, changed-input graph replay, arithmetic, image OCR/colors/shapes, four-image order and serving stability. The throughput suite uses short prompts and has no broad task-quality scoring. Its 86 output-length differences against Tony's reference neither prove a quality loss nor prove equivalence. The 93,335-token prefill case requests one output token, so it measures prefill performance rather than long-context answer quality.

## Options that can trade speed for quality or capability

| Option | Status here | Potential benefit | What must be protected |
|---|---|---|---|
| FP8 activations in the remaining `wo_a` projection | Proposed; not enabled | Faster projection using an existing FP8 matrix kernel | The current fallback keeps activations in BF16. Lowering their precision can alter logits and answers; numerical checks and task-quality evaluation are required. |
| More aggressive weight or KV-cache quantization | No additional reduction applied | Less memory and potentially faster execution | Accuracy, especially on sensitive reasoning and long-context tasks. Kernel support, memory savings or a successful boot alone do not establish an acceptable tradeoff. |
| Thinking/reasoning disabled | Existing benchmark and server default | Less reasoning work and potentially shorter responses | Complex reasoning performance. The reported headline results are **thinking off**, not measurements of the model's full reasoning mode. |
| Reduced context limits, shorter output caps or disabled vision | Not adopted as part of the optimization; benchmarks use fixed short output budgets | Lower resource use or shorter request time | Usable context, answer completeness and multimodal capability. Report these as changes to what the service can do, not simply as speed improvements. |

The benchmark uses thinking off to match the community reference. A request can select thinking with `chat_template_kwargs: {"thinking": true}` on the configured API. Performance and quality in that mode have not been measured here, so the published throughput should not be assumed to apply to it.

## Speculative decoding needs a separate check

A shorter draft window with the same correct target-verification procedure is not inherently a lower-quality model setting. The proposed k=3/4/5 sweep is intended to reduce wasted work, not to weaken verification. Nevertheless, this stack's exact draft, rejection, graph and sampling configuration must be checked against a target-model reference; changing settings is not sufficient evidence of equivalent behavior. Relaxing verification to accept more drafts would be a different, quality-sensitive experiment and should be labeled accordingly.

## How to judge a proposed change

Establish the quality criteria before using throughput to choose a winner. Compare the same prompts, model revision, reasoning mode, context/output limits, tools and decoding settings, with task coverage that represents the intended use. Include code or task correctness, reasoning accuracy, structured outputs and tool arguments, relevant image tasks, and long-context answers where those capabilities matter. Numerical kernel tests support this evaluation but do not replace it.

Measure latency, aggregate throughput, completed useful work and failure rates alongside quality. A change that emits more tokens while solving fewer tasks is not an improvement. Use repeated runs where outputs or timing vary, distinguish semantic correctness from exact text equality, and retain the stronger configuration if the measured quality loss outweighs the speed benefit. Any accepted capability tradeoff should be explicit and justified for its workload.

The current deployment has bounded numerical and functional validation, **not a completed broad model-quality evaluation**. Until that gap is addressed, further precision changes remain experimental. [Validation coverage](validation.md) · [Optimization results and next experiments](mia-improvements.md) · [Benchmark method](benchmark-method.md).
