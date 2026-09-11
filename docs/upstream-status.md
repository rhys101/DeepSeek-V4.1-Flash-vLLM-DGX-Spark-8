# Upstream repository review — 11 September 2026

**Tony has two commits after our tested recipe pin; neither changes the main benchmark, model kernels or build recipe.** Mia's head remains the revision already reviewed. This check updates comparison documentation and does not change the live deployment.

## Tony

Reviewed head: [`458fadec6106e6fd84f7eda36dd6e0a8fa219ef6`](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/458fadec6106e6fd84f7eda36dd6e0a8fa219ef6). Our tested recipe and original benchmark pin remain `ca662ac35193c69ace9cee37f13a94abf2eff0fc`.

| Commit | Date (UTC) | Change |
|---|---|---|
| [`592540c`](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/commit/592540c69853a8ce9285236ebfd6e54dfc83a013) | 2026-09-10 19:28 | Vision and tool calling: README section, end-to-end checks, recipe note |
| [`458fade`](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/commit/458fadec6106e6fd84f7eda36dd6e0a8fa219ef6) | 2026-09-11 02:29 | One-command restore of boot 10 (used after a worker power-off) |

The vision/tool addition supplies a seven-case harness: three image checks and four tool-calling checks covering argument generation, a supplied tool-result round trip, parallel calls and forced tool choice. His recorded run passed all seven. [Recorded results](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/458fadec6106e6fd84f7eda36dd6e0a8fa219ef6/results/boot10/vision-tools.txt).

The restore addition checks worker reachability, model mounts, local Engram copies, staged-file identity and GPU throughput before relaunching. It restores temporary launch files, starts workers before the head and runs post-launch checks. The recorded recovery followed an accidental worker shutdown. These scripts contain Tony's fleet-specific users, paths and addresses; they are useful operational references rather than drop-in eight-node scripts. The current repository already checks image/config identity, uses local checkpoints, launches workers first and waits for inference readiness. A dedicated recovery command and fuller tool-calling coverage remain possible follow-ups.

The [restored-boot probes](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/458fadec6106e6fd84f7eda36dd6e0a8fa219ef6/results/boot10/restore-20260911/idletest.txt) report 90.3–91.9 tok/s counting decode and 71.6–73.2 tok/s coding decode. They use short idle/back-to-back checks. They do not replace the existing C1–C6 and cold-prefill benchmark, which is byte-for-byte unchanged. The restore also reports a different KV capacity under the same serving configuration; this is a fresh startup observation, not a new model/kernel optimization.

The diff contains no changes under `build/`, `patch/`, `launch/` or `bench/`; only documentation/results and operational tools changed. The SHA-256 of the current boot 10 benchmark remains `e352b099e7f61844cf7144a3df62db978a2e8f74a2cf553aaddbce50b09a5715`. Script, prompt-set and reporter hashes also match our pinned copies. [Machine-readable check](../results/reference/upstream-check-2026-09-11.json).

## Mia

Reviewed head: [`e59e6eb67479aa68f6fa700c600dc90a0729b5ec`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/tree/e59e6eb67479aa68f6fa700c600dc90a0729b5ec), unchanged since the optimization review. Published performance figures belong to the three-Spark SGLang deployment. The four-Spark profile has configuration/script checks but no recorded boot at this revision.

| Prose measurement (tok/s) | Mia: 3 Sparks, SGLang | Tony: 4 Sparks, vLLM | This deployment: 8 Sparks, vLLM |
|---|---|---|---|
| C1 decode per stream | 37.9 | 24.37 | 41.28 |
| C4 decode per stream | 20.9 | 18.37 | 26.52 |
| C4 aggregate | 78.6 | 69.94 | 93.45 |

The Mia and community prose workloads have not been established to match. [Complete prose tables and comparison limits](comparison.md#prose-comparison-with-mias-three-sparks) · [Reported values and source hash](../results/reference/mia-reported-prose.json).
