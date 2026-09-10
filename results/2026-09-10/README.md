# C1–C6, prefill and deployment validation

`bench-spark8.json` contains 54 measured category batches / 189 requests and four cold-prefill cases. `bench-tony-boot10.json` is the pinned four-Spark reference. All category and prefill input counts match; 82 category output lengths differ. The maximum measured prompt is 93,335 tokens.

Source, package, image and health manifests identify the deployment. Image OCR/color/shape and four-image ordering checks passed. The raw image responses include a correctly identified image returned as nested objects and the explicit string-schema smoke check. These are limited API smoke checks, not a model-quality evaluation.

The GPU-state probe ran with the model unloaded and is separate from the serving benchmark. See [validation coverage](../../docs/validation.md), [comparison](../../docs/comparison.md) and [methodology](../../docs/benchmark-method.md).
