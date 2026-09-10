#!/usr/bin/env python3
"""Detailed tables from a v41bench.py JSON: per category x concurrency.
  decode   = per-stream tok/s after the first token (mean over the streams)
  TTFT     = time to first token, mean over the streams (s)
  e2e      = per-stream tok/s over the whole request, TTFT included (mean)
  agg      = aggregate throughput: all streams' tokens / batch wall time
plus the cold-prefill sweep. usage: bench_report.py BENCH.json [--vs OTHER.json]"""
import json
import statistics as st
import sys

d = json.load(open(sys.argv[1]))
vs = json.load(open(sys.argv[sys.argv.index("--vs") + 1])) if "--vs" in sys.argv else None
levels = sorted({b["c"] for b in d["batches"]})
order = ["coding", "json", "math", "reasoning", "format", "summary", "prose", "narrative", "ceiling_count"]
cats = [c for c in order if any(b["category"] == c for b in d["batches"])]


def cell(data, c, cat):
    return next((b for b in data["batches"] if b["c"] == c and b["category"] == cat), None)


def metric(b, m):
    if b is None:
        return None
    rs = b["requests"]
    if m == "decode":
        v = [r["decode_tok_s"] for r in rs if r["decode_tok_s"]]
    elif m == "ttft":
        v = [r["ttft_s"] for r in rs if r["ttft_s"] is not None]
    elif m == "e2e":
        v = [r["completion_tokens"] / r["total_s"] for r in rs if r["total_s"]]
    else:
        return b["agg_tok_s"]
    return st.mean(v) if v else None


def table(title, m, fmt, data=d):
    out = [f"### {title}", "", "| category | " + " | ".join(f"C{c}" for c in levels) + " |",
           "|---|" + "---|" * len(levels)]
    for cat in cats:
        vals = [metric(cell(data, c, cat), m) for c in levels]
        name = "counting (ceiling)" if cat == "ceiling_count" else cat
        out.append(f"| {name} | " + " | ".join("" if v is None else fmt.format(v) for v in vals) + " |")
    return "\n".join(out)


print(f"## {d['label']} ({d['started']} to {d.get('finished', '?')})\n")
print(f"{d.get('notes', '')}\n")
print("Prompt set `" + d["prompt_set"] + "`, temperature 0, thinking off, streaming; one batch per cell "
      "(C streams released together). Short prompts (about 30-120 tokens), 150-256 token budgets.\n")
print(table("Decode: per-stream tok/s after the first token", "decode", "{:.1f}") + "\n")
print(table("Aggregate throughput: tok/s across all streams (wall time, TTFT included)", "agg", "{:.1f}") + "\n")
print(table("TTFT: mean time to first token (s)", "ttft", "{:.2f}") + "\n")
print(table("End-to-end per stream: tok/s over the whole request, TTFT included", "e2e", "{:.1f}") + "\n")
if vs:
    rows = ["### Decode per stream vs " + vs["label"], "", "| category | " + " | ".join(f"C{c}" for c in levels) + " |",
            "|---|" + "---|" * len(levels)]
    for cat in cats:
        vals = []
        for c in levels:
            a, b = metric(cell(d, c, cat), "decode"), metric(cell(vs, c, cat), "decode")
            vals.append("" if not a or not b else f"{b:.1f} → {a:.1f}")
        rows.append(f"| {'counting (ceiling)' if cat == 'ceiling_count' else cat} | " + " | ".join(vals) + " |")
    print("\n".join(rows) + "\n")
if d.get("prefill"):
    print("### Cold prefill (unique prompt, 1-token reply; TTFT = whole prefill)\n")
    print("| target | prompt tokens | TTFT (s) | prefill tok/s |\n|---|---|---|---|")
    for p in d["prefill"]:
        print(f"| {p['target'] // 1000}K | {p['prompt_tokens']:,} | {p['ttft_s']:.2f} | {p['prefill_tok_s']:,.0f} |")
