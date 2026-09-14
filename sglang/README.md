# SGLang on eight DGX Sparks

This directory contains the source, configuration, launcher and benchmark
evidence for the SGLang deployment of DeepSeek V4.1 Flash.

- [Current performance and hardware setup](../README.md)
- [Deployment guide: build, validate, serve and operate](../docs/getting-started.md)
- [Progress and historical benchmarks](../docs/progress.md)
- [SG18 native-prefill source and integration settings](experiments/sg18-prefill-tp-split/)
- [SG18 measurements and quality limits](docs/sg18-prefill-results.md)
- [Pinned source and build identity](docs/build-and-pins.md)

The standard Dockerfile and `scripts/cluster.py` package **SG5**. The measured
**SG18** build uses the separate integration linked above; changing the SG5
example JSON alone does not install it.

Run the deployment guide’s commands from this directory. Start configuration
with `configs/cluster.example.json`; keep machine-specific values in the
gitignored `configs/cluster.local.json`.

See [NOTICE](NOTICE) for the Mia-derived AGPL adaptation, Apache-2.0 SGLang
source and MIT benchmark material, and the accompanying license texts.
