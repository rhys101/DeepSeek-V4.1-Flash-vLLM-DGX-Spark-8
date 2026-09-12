// One explicit source edit for C64/C128; pinned upstream files remain untouched.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';

export function createConcurrencyRuntime(upstreamRoot, runtimeParent, lock) {
  const filename = 'server/collectors/DecodeBench.js';
  const original = fs.readFileSync(path.join(upstreamRoot, filename), 'utf8');
  const sha = text => createHash('sha256').update(text).digest('hex');
  if (sha(original) !== lock.files_sha256[filename]) throw Error('DecodeBench source does not match the pinned upstream hash');
  const before = 'const ALLOWED_CONCURRENCIES = new Set([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32]);';
  const after = 'const ALLOWED_CONCURRENCIES = new Set([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 64, 128]);';
  if (original.split(before).length !== 2) throw Error('Expected exactly one pinned concurrency allowlist');
  const modified = original.replace(before, after);
  fs.mkdirSync(runtimeParent, { recursive: true });
  const root = fs.mkdtempSync(path.join(runtimeParent, 'concurrency-'));
  for (const name of Object.keys(lock.files_sha256)) {
    const source = path.join(upstreamRoot, name), destination = path.join(root, name);
    if (sha(fs.readFileSync(source)) !== lock.files_sha256[name]) throw Error(`Changed upstream file: ${name}`);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.copyFileSync(source, destination);
  }
  fs.writeFileSync(path.join(root, filename), modified);
  return {
    root,
    patch: { file: filename, before_sha256: sha(original), after_sha256: sha(modified),
      before, after, added_concurrencies: [64, 128],
      description: 'Only the allowed concurrency set is extended. Prompts, token budgets, sampling, warmup, request execution and metric calculations are unchanged.' },
    cleanup() { fs.rmSync(root, { recursive: true, force: true }); },
  };
}
