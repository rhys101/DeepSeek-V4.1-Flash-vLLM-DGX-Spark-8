/**
 * Prefill-bench context sizes — shared by the Node runner and the React dialog.
 */

export const PREFILL_CONTEXT_SIZES = [
  1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 300000,
];

export const PREFILL_DEFAULT_CONTEXT_SIZES = [4096, 8192, 16384, 32768];

const SIZE_LABELS = new Map([
  [1024, "1k"],
  [2048, "2k"],
  [4096, "4k"],
  [8192, "8k"],
  [16384, "16k"],
  [32768, "32k"],
  [65536, "64k"],
  [131072, "128k"],
  [262144, "256k"],
  [300000, "300k"],
]);

/** @param {number} tokens */
export function formatContextSize(tokens) {
  const n = Number(tokens);
  if (SIZE_LABELS.has(n)) return SIZE_LABELS.get(n);
  if (!Number.isFinite(n) || n <= 0) return String(tokens);
  if (n >= 1000 && n % 1000 === 0) return `${n / 1000}k`;
  return String(n);
}
