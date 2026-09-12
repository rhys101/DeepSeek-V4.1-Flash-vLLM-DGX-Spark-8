// CLI around pinned sparkDash functions, with an explicit optional C64/C128 allowlist extension.
import fs from 'node:fs';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { parseArgs } from 'node:util';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createConcurrencyRuntime } from './concurrency-extension.mjs';

const { values: a } = parseArgs({ options: {
  base: { type: 'string' }, model: { type: 'string', default: 'deepseek-v41-flash' },
  out: { type: 'string' }, label: { type: 'string' },
  trials: { type: 'string', default: '2' },
  sizes: { type: 'string', default: '4096,16384,32768,65536,131072' },
  levels: { type: 'string', default: '1,2,3,4,6,8' },
  'prefill-only': { type: 'boolean', default: false },
  'extended-concurrency': { type: 'boolean', default: false },
} });
if (!a.base || !a.out || !a.label) throw Error('--base URL --out NEW_DIRECTORY --label LABEL required');
const url = new URL(a.base);
if (!['http:', 'https:'].includes(url.protocol) || !['/', '/v1', '/v1/'].includes(url.pathname)) throw Error('Use an HTTP(S) endpoint root or /v1');
const trials = Number(a.trials), sizes = a.sizes.split(',').map(Number), levels = a.levels.split(',').map(Number);
if (!Number.isInteger(trials) || trials < 1 || trials > 10) throw Error('trials must be 1..10');
const root = path.resolve(a.out);
if (fs.existsSync(root)) throw Error('Output directory exists; choose a fresh path');
fs.mkdirSync(root, { recursive: true });
for (const [env, file] of Object.entries({
  BENCH_HISTORY_PATH: 'decode-history.json', BENCH_ACTIVE_PATH: 'decode-active.json',
  PREFILL_BENCH_HISTORY_PATH: 'prefill-history.json', PREFILL_BENCH_ACTIVE_PATH: 'prefill-active.json',
})) process.env[env] = path.join(root, file);

const pre = await import('./upstream/server/collectors/PrefillBench.js');
const stream = await import('./upstream/server/collectors/LlmStreaming.js');
if (pre.normalizeContextSizes(sizes).length !== sizes.length) throw Error('Invalid or duplicate context size');
const lock = JSON.parse(fs.readFileSync(new URL('./upstream.lock.json', import.meta.url)));
for (const [name, hash] of Object.entries(lock.files_sha256)) {
  const data = fs.readFileSync(new URL(`./upstream/${name}`, import.meta.url));
  if (createHash('sha256').update(data).digest('hex') !== hash) throw Error(`Changed upstream file: ${name}`);
}
const extension = a['extended-concurrency'] ? createConcurrencyRuntime(
  fileURLToPath(new URL('./upstream/', import.meta.url)),
  fileURLToPath(new URL('./.local/', import.meta.url)), lock) : null;
const decodeModule = await import(extension
  ? pathToFileURL(path.join(extension.root, 'server/collectors/DecodeBench.js')).href
  : './upstream/server/collectors/DecodeBench.js');
const { decodeBenchManager: decode, DECODE_BENCH_DEFAULTS: defaults } = decodeModule;
if (levels.length !== new Set(levels).size || levels.some(c => !defaults.allowedConcurrencies.includes(c))) {
  extension?.cleanup();
  throw Error(`Invalid concurrency level; allowed levels: ${defaults.allowedConcurrencies.join(',')}. Use --extended-concurrency for C64/C128.`);
}
const source = extension ? { ...lock, unchanged_upstream_files: false,
  pinned_upstream_files_verified: true, local_changes: [extension.patch] } : lock;
const result = { status: 'running', label: a.label, started: new Date().toISOString(),
  source, node: process.version, config: { ...a }, prefill: [], decode: [],
  method: 'Pinned sparkDash prompts and streaming metrics. Prefill C1, fresh UUID prefix per request, 8-token budget, temperature 0, thinking off. Decode prose, 256 tokens, ascending concurrency per trial; original prompt reuse and warmup. Client timing, no hardware polling. Usage tokens required. Tweet source version unconfirmed.' };
function save() { fs.writeFileSync(path.join(root, 'result.json'), JSON.stringify(result, null, 2) + '\n'); }
let activeJob = null;
const stop = new AbortController();
process.on('SIGINT', () => { stop.abort(); if (activeJob) decode.cancel('comparison', activeJob); process.exitCode = 130; });
process.on('exit', () => extension?.cleanup());

async function prefill(target, trial, warmup = false) {
  const salt = randomUUID(), prompt = pre.buildPrefillPrompt(target, salt);
  const body = { model: a.model, messages: [{ role: 'user', content: prompt }],
    max_tokens: 8, temperature: 0, top_p: 1, stream: true, stream_options: { include_usage: true } };
  stream.applyThinkingFlags(body, a.model, false);
  const request = await stream.runStreamingRequest(`${url.origin}/v1/chat/completions`, body,
    AbortSignal.any([stop.signal, AbortSignal.timeout(pre.timeoutMsForSize(target))]),
    { retryOnThinking400: true, thinking: false, debug: true, apiKey: process.env.OPENAI_API_KEY || null });
  if (request.error || !request.usage?.promptTokens || request.ttftMs <= 0) throw Error(`Prefill ${target}: ${request.error || 'missing usage / TTFT'}`);
  if (!warmup) {
    const row = { trial, targetTokens: target, salt, promptChars: prompt.length,
      promptSha256: createHash('sha256').update(prompt).digest('hex'), request };
    result.prefill.push(row); save();
    console.log(JSON.stringify({ phase: 'prefill', trial, target, promptTokens: request.prefillTokens, ttftMs: request.ttftMs, prefillTps: request.prefillTps }));
  }
}

try {
  save();
  for (let trial = 1; trial <= trials; trial++) {
    await prefill(512, trial, true);
    for (const size of sizes) await prefill(size, trial);
    if (a['prefill-only']) continue;
    const start = decode.start({ sparkId: 'comparison', lanIp: url.hostname,
      port: Number(url.port || (url.protocol === 'https:' ? 443 : 80)), tls: url.protocol === 'https:',
      modelId: a.model, concurrencies: levels, maxTokens: 256, promptType: 'prose', debug: true,
      apiKey: process.env.OPENAI_API_KEY || null });
    activeJob = start.benchId;
    let job = start, lastCompleted = -1;
    while (job.status === 'running') {
      await delay(500); job = decode.getJob(activeJob);
      if (job.progress.completedLevels !== lastCompleted) {
        lastCompleted = job.progress.completedLevels;
        console.log(JSON.stringify({ phase: 'decode', trial, progress: job.progress }));
      }
    }
    activeJob = null;
    result.decode.push({ trial, job }); save();
    if (job.status !== 'completed' || job.results.length !== levels.length || job.results.some(w => w.error || w.streams.some(s => !s.usage?.promptTokens || !s.usage?.completionTokens || s.completionTokens !== 256))) throw Error('Decode job failed or missing usage / 256-token completion');
  }
  result.status = 'PASS'; result.finished = new Date().toISOString(); save();
  console.log(JSON.stringify({ status: result.status, output: root }));
} catch (error) {
  result.status = 'FAIL'; result.error = String(error); save(); throw error;
} finally {
  extension?.cleanup();
}
