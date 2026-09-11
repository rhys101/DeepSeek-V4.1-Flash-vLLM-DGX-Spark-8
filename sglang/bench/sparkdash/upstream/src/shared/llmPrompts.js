/**
 * Shared LLM prompt catalogs (Showcase + Decode bench).
 * Structural = code/data formats; Text = prose (no code).
 */

export const TEXT_PROMPTS = [
  "Write a clear essay explaining unified memory on NVIDIA GB10 Sparks for a technical but non-specialist reader. Keep expanding with examples and analogies.",
  "Write a vivid sci-fi scene set in a liquid-cooled server room at 3 a.m. Keep expanding the scene with sensory detail and dialogue.",
  "Write a pirate-captain monologue explaining KV-cache pressure and prefill vs decode to the crew. Keep expanding with more shanties and metaphors.",
  "Write a nursery-rhyme style poem about thermal throttling and power caps. Add many stanzas and keep going.",
  "Write naturalistic dialogue between two ops engineers debugging a stuck vLLM queue. Continue for many turns without wrapping up.",
  "Write a courtroom cross-examination where the witness is a tokenizer. Keep adding Q&A exchanges.",
  "Write a travel-brochure parody for visiting a liquid-cooled GPU rack. Flowery marketing tone; keep expanding sections.",
  "Write a radio weather report for a GPU cluster: temperature fronts across racks, token-storm warnings. Keep broadcasting.",
  "Write packaging copy for a fictional energy drink called Prefill Punch aimed at LLM operators. Expand with flavors, warnings, and testimonials.",
  "Write chapter 1 of a short story titled \"The Day the Slots Went to Zero,\" then keep expanding the narrative without ending.",
  "Write a lecture transcript on TTFT, ITL, and e2e latency for inference operators. Keep teaching with more examples.",
  "Write a memoir-style recollection of the first time a cluster OOMed mid-demo. Keep expanding with flashbacks and lessons.",
  "Write a sports-commentator style play-by-play of concurrent decode waves hitting a 4-GPU node. Keep calling the action.",
  "Write a bedtime story for SREs about a friendly KV cache that grew too large. Keep adding chapters.",
  "Write an op-ed arguing that tok/s is overrated without TTFT context. Expand with rebuttals and counter-rebuttals.",
  "Write a campfire story told by a retired load balancer about the great token flood of '27. Keep going.",
];

export const STRUCTURAL_PROMPTS = [
  "Emit only a JSON array of fake GPU metrics rows. Each object needs host, gpuIndex, utilPct, tempC, powerW, memUsedMb. Invent many rows. No markdown. Keep expanding the array.",
  "Emit only an HTML FAQ about CUDA and vLLM. Use <h2> and <p> for many Q&A pairs. No markdown fences. Keep adding sections.",
  "Emit a Markdown comparison table: llama.cpp vs vLLM vs SGLang. Columns: feature, llama.cpp, vLLM, SGLang. Fill many rows and keep adding.",
  "Stream a fake syslog of cluster events (timestamps, INFO/WARN/ERROR, services). Keep lines coming continuously.",
  "Write only valid YAML for a multi-service docker-compose stack with redis, postgres, api, and worker. Expand heavily with env, volumes, and healthchecks.",
  "Emit a CSV of invented datacenter PUE readings: date,site,pue,itKw,facilityKw. Many rows. CSV only, no commentary. Keep adding rows.",
  "Generate a GraphQL schema as SDL only: types Query, Mutation, User, Job, Metric. Add many fields, enums, and interfaces. Keep expanding.",
  "Generate an OpenAPI 3 paths snippet as JSON for /v1/models and /v1/chat/completions. Expand schemas heavily. JSON only.",
  "Emit a Markdown cheatsheet: nvidia-smi flags vs what they show. Dense table, many rows. Keep adding.",
  "Generate a long TOML config for a fictional inference gateway: listeners, routes, retries, budgets. Keep expanding sections.",
  "Emit only SQL: CREATE TABLE + many INSERT statements for gpu_jobs(id, host, model, tokens, ms). Keep inserting.",
  "Generate a Mermaid sequenceDiagram (fenced) for client → proxy → vLLM → GPU. Expand with retries, queues, and metrics spans.",
  "Emit a long alphabetized Markdown definition list glossary of ML-systems jargon (KV cache, TTFT, ITL, MTP, …). Keep adding terms.",
  "Generate only Rust-flavored pseudocode for a lock-free token ring buffer. Keep expanding with more functions and tests.",
  "Emit a fake Prometheus text exposition dump for showcase_tokens_total, showcase_ttft_seconds, and related series. Keep adding metrics.",
  "Emit only a Python module of dataclasses for SparkHost, GpuSlice, and LlmEndpoint with typed fields, validators, and docstrings. Keep expanding.",
  "Write only valid JSON (no markdown). Generate OpenAPI-style paths as JSON: paths{}, components.schemas{}. Invent many endpoints and schemas.",
  "List 40 shell one-liners useful for NVIDIA Sparks / DGX. Commands only, one per line, no commentary. Then invent more variants.",
];

export const FILL_TO_MAX_SUFFIX =
  " Continue generating until you hit the maximum output length; do not stop early—keep expanding with more content.";

/** Same prompt as glm-5.3-flash-sm120 `tests/bench_decode.py --structured`. */
export const DECODE_STRUCTURED_PROMPT =
  "Count from 1 to 200. Output only the numbers, separated by spaces. No other text.";

/** Same prompt as glm-5.3-flash-sm120 `tests/bench_decode.py` default (hash-map prose). */
export const DECODE_PROSE_PROMPT =
  "Write a detailed step-by-step explanation of how a hash map works, " +
  "including collision handling, resizing, and time complexity. Be thorough.";

/** High-accept code: repeated identical-shape helpers (not an essay with `def`). */
export const DECODE_CODE_PROMPT =
  "Output only Python source code. No comments, no docstrings, no markdown fences. " +
  "Write functions clamp_00 through clamp_49. Each function is exactly:\n" +
  "def clamp_NN(x, lo=0, hi=1):\n" +
  "    if x < lo:\n" +
  "        return lo\n" +
  "    if x > hi:\n" +
  "        return hi\n" +
  "    return x\n" +
  "Change only the function name suffix (00, 01, … 49). One blank line between functions. No other text.";

/**
 * JSON/YAML-ish catalog (Showcase structural #0). Labels an output shape only —
 * do not send response_format / grammars / guided JSON with this prompt.
 */
export const DECODE_JSON_PROMPT = STRUCTURAL_PROMPTS[0];

/** Decode-bench output types (not guided decoding). Structured is the default. */
export const DECODE_BENCH_TYPES = ["structured", "prose", "code", "json"];
export const DECODE_BENCH_DEFAULT_TYPE = "structured";

export const DECODE_BENCH_PROMPTS = {
  structured: DECODE_STRUCTURED_PROMPT,
  prose: DECODE_PROSE_PROMPT,
  code: DECODE_CODE_PROMPT,
  json: DECODE_JSON_PROMPT,
};

export const DECODE_BENCH_TYPE_META = [
  {
    id: "structured",
    label: "Structured",
    hint: "Count 1→200, numbers only — lab structured protocol",
  },
  {
    id: "prose",
    label: "Prose",
    hint: "Hash-map explanation — lab default bench prompt",
  },
  {
    id: "code",
    label: "Code",
    hint: "clamp_00…clamp_49 Python helpers — code-shaped, no comments",
  },
  {
    id: "json",
    label: "JSON",
    hint: "JSON GPU-metrics catalog — output type only, not guided JSON",
  },
];

/**
 * @param {unknown} type
 * @returns {"structured" | "prose" | "code" | "json"}
 */
export function normalizeDecodeBenchType(type) {
  const t = String(type || "").trim().toLowerCase();
  return DECODE_BENCH_TYPES.includes(t) ? t : DECODE_BENCH_DEFAULT_TYPE;
}

/**
 * @param {unknown} type
 * @returns {string}
 */
export function decodeBenchPromptForType(type) {
  return DECODE_BENCH_PROMPTS[normalizeDecodeBenchType(type)];
}

/**
 * @param {unknown} type
 * @returns {string}
 */
export function decodeBenchTypeLabel(type) {
  const id = normalizeDecodeBenchType(type);
  return DECODE_BENCH_TYPE_META.find((m) => m.id === id)?.label || "Structured";
}

/**
 * Append a hard fill-to-max instruction unless the prompt already states it.
 * Soft phrases like "keep expanding" alone do not skip — models still EOS early.
 * @param {string} prompt
 */
export function withFillToMaxInstruction(prompt) {
  const p = String(prompt || "").trim();
  if (!p) return p;
  if (
    /maximum output length|do not stop early|until you hit the (maximum|output)/i.test(
      p
    )
  ) {
    return p;
  }
  return `${p}${FILL_TO_MAX_SUFFIX}`;
}

/**
 * Build N prompts for a prompt type. Mixed interleaves structural then text.
 * @param {"structural" | "text" | "mixed"} type
 * @param {number} count
 * @returns {string[]}
 */
export function pickShowcasePrompts(type, count) {
  const n = Math.max(1, Math.floor(count));
  if (type === "text") return takeCycled(TEXT_PROMPTS, n);
  if (type === "structural") return takeCycled(STRUCTURAL_PROMPTS, n);

  const out = [];
  let si = 0;
  let ti = 0;
  for (let i = 0; i < n; i++) {
    if (i % 2 === 0) {
      out.push(STRUCTURAL_PROMPTS[si % STRUCTURAL_PROMPTS.length]);
      si += 1;
    } else {
      out.push(TEXT_PROMPTS[ti % TEXT_PROMPTS.length]);
      ti += 1;
    }
  }
  return out;
}

/**
 * Decode-bench prompts for a workload type. Concurrent streams get a unique
 * suffix so they do not share a prefix-cache block; C1 is the exact prompt.
 * @param {number} count
 * @param {unknown} [type]
 * @returns {string[]}
 */
export function pickDecodeBenchPrompts(count, type) {
  const base = decodeBenchPromptForType(type);
  const n = Math.max(1, Math.floor(count));
  if (n <= 1) return [base];
  const out = [];
  for (let i = 0; i < n; i++) {
    out.push(`${base} (stream ${i + 1}/${n})`);
  }
  return out;
}

/**
 * @param {string[]} pool
 * @param {number} n
 */
function takeCycled(pool, n) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(pool[i % pool.length]);
  return out;
}
