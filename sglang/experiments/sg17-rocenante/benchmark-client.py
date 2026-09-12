"""Repeat the pinned coding/prose C1 workload, with C4/C8 controls."""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--label', required=True)
ap.add_argument('--base', required=True, help='OpenAI-compatible /v1 endpoint')
a = ap.parse_args()
p = Path(a.out)
p.mkdir(parents=True, exist_ok=False)
base = a.base.rstrip('/')
repo = Path(__file__).resolve().parents[3]
kit = repo / 'sglang'
bench = repo / 'bench/v41bench.py'

def get(path):
    with urllib.request.urlopen(base.removesuffix('/v1') + path, timeout=20) as response:
        return json.load(response)

def save(name, value):
    (p / name).write_text(json.dumps(value, indent=2) + '\n')

before = get('/v1/loads')
assert all(x['num_running_reqs'] == x['num_waiting_reqs'] == 0 for x in before['loads'])
info = get('/server_info')
assert (info['tp_size'], info['ep_size'], info['context_length'], info['max_total_tokens'], info['max_running_requests'], info['speculative_dspark_block_size'], info['chunked_prefill_size']) == (8, 4, 1000000, 8000000, 128, 5, 2048)
save('loads-before.json', before)
save('server-info.json', info)
want = 'e0d6b2d25bd585d11fbdf39c2ddcdf7a4de8ab685af6bd42465e69f3ee6e80a8'
assert hashlib.sha256(bench.read_bytes()).hexdigest() == want
spec = importlib.util.spec_from_file_location('v41bench', bench)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
args = SimpleNamespace(base=base, model='deepseek-v41-flash')
for category, prompt, _ in module.CATEGORIES[:3]:
    module.run_batch(args, 1, category, prompt, 64, 'warm')
category, prompt, budget = module.CATEGORIES[0]
rows = []
# Excluded first coding batch at each concurrency establishes the same reused
# prompt state on both an existing baseline process and a fresh candidate.
warmups = [module.run_batch(args, c, category, prompt, budget, 'run') for c in [1, 4, 8]]
for trial in range(1, 6):
    levels = ([1, 4, 8] if trial % 2 else [8, 4, 1]) if trial <= 3 else [1]
    for concurrency in levels:
        row = module.run_batch(args, concurrency, category, prompt, budget, 'run')
        row['trial'] = trial
        assert all(x['completion_tokens'] == 200 and x['prompt_tokens'] > 0 for x in row['requests'])
        rows.append(row)
        print(json.dumps({'phase': 'coding', 'trial': trial, 'c': concurrency, 'decode': row['per_stream_tok_s'], 'aggregate': row['agg_tok_s'], 'ttft': row['ttft_mean_s']}), flush=True)
save('coding.json', {'label': a.label, 'source_sha256': want, 'batches': rows, 'excluded_coding_warmups': warmups, 'method': 'Pinned benchmark; standard warmups plus one excluded full coding batch per C. Reused identical prefixes. Five C1 and three C4/C8 waves; C1/C4/C8 ordering reversed on trial 2. No cache flush.'})
subprocess.run(['node', str(kit / 'bench/sparkdash/run.mjs'), '--base', base, '--model', 'deepseek-v41-flash', '--out', str(p / 'sparkdash'), '--label', a.label, '--trials', '3', '--sizes', '4096', '--levels', '1,4,8'], check=True, timeout=300)
after = get('/v1/loads')
save('loads-after.json', after)
assert all(x['num_running_reqs'] == x['num_waiting_reqs'] == 0 for x in after['loads'])
save('completion.json', {'status': 'PASS', 'finished': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'label': a.label})
print('BENCHMARK_COMPLETE', flush=True)
