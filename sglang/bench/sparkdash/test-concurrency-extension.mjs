import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createConcurrencyRuntime } from './concurrency-extension.mjs';
import { DECODE_BENCH_DEFAULTS as originalDefaults } from './upstream/server/collectors/DecodeBench.js';

const upstream = fileURLToPath(new URL('./upstream/', import.meta.url));
const parent = fileURLToPath(new URL('./.local/', import.meta.url));
const lock = JSON.parse(fs.readFileSync(new URL('./upstream.lock.json', import.meta.url)));

test('extension adds only C64/C128 and leaves every other source byte unchanged', async () => {
  const runtime = createConcurrencyRuntime(upstream, parent, lock);
  try {
    for (const name of Object.keys(lock.files_sha256)) {
      const before = fs.readFileSync(path.join(upstream, name), 'utf8');
      const after = fs.readFileSync(path.join(runtime.root, name), 'utf8');
      assert.equal(after, name === runtime.patch.file ? before.replace(runtime.patch.before, runtime.patch.after) : before);
    }
    const { DECODE_BENCH_DEFAULTS: extended } = await import(pathToFileURL(path.join(runtime.root, runtime.patch.file)).href);
    assert.deepEqual(extended.allowedConcurrencies, [...originalDefaults.allowedConcurrencies, 64, 128]);
    assert.deepEqual({ ...extended, allowedConcurrencies: [] }, { ...originalDefaults, allowedConcurrencies: [] });
    assert.equal(originalDefaults.allowedConcurrencies.includes(64), false);
    assert.equal(originalDefaults.allowedConcurrencies.includes(128), false);
  } finally {
    runtime.cleanup();
  }
});

test('extension refuses an unpinned DecodeBench source', () => {
  const changed = structuredClone(lock);
  changed.files_sha256['server/collectors/DecodeBench.js'] = '0'.repeat(64);
  assert.throws(() => createConcurrencyRuntime(upstream, parent, changed), /pinned upstream hash/);
});
