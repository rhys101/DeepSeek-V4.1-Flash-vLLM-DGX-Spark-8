"""Verify the promotion against immutable timing and capacity evidence offline."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

if sys.flags.optimize:
    raise RuntimeError('Run without -O: recorded validators use assertions.')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = REPO / 'sglang/results/sg18-prefill-tp-split'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
promotion = read(RESULTS / 'promotion.json')
assert promotion['status'] == 'PROMOTED'
for name, digest in promotion['evidence_sha256'].items():
    assert sha(RESULTS / name) == digest, name
source = read(HERE / 'source-manifest.json')
assert source['status'] == 'PROMOTED_AFTER_CAPACITY_RETEST'
assert source['source_files'] == promotion['source_files']
assert all(sha(HERE / 'source' / name) == digest for name, digest in source['source_files'].items())
historical = read(RESULTS / 'measurements.json')
assert promotion['cold_cells'] == historical['cold_runs'][promotion['cold_run']]['validation']['measured_cells']
assert promotion['readme_runs'] == historical['readme_runs'][promotion['readme_run']]['runs']
retest = read(RESULTS / 'capacity-retest.json')
assert promotion['capacity'] == retest['capacity_validation']
assert promotion['guard'] == retest['raw_guard_validation']
assert promotion['primary_long_prefill_time_reductions_percent'] == retest['earlier_confirmation_time_reductions_percent']
health = read(RESULTS / 'promotion-health.json')
assert health['status'] == 'PASS' and all(not jobs for jobs in health['active_experiment_jobs'].values())
assert len(health['ranks']) == 8
for rank in health['ranks']:
    assert rank['state']['Running'] and not rank['state']['OOMKilled']
    assert rank['candidate'] == f'sglang8-sparkkernels-prefillnative01-r{rank["rank"]}'
    for name, digest in source['source_files'].items():
        assert rank['source_and_cache_hashes']['/sgl-workspace/sglang/' + name] == digest
with tempfile.TemporaryDirectory(prefix='prefill-promotion-verify-') as folder:
    root = Path(folder)
    with zipfile.ZipFile(RESULTS / 'capacity-retest-evidence.zip') as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        for name in archive.namelist():
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts, name
            data = archive.read(name)
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    assert read(root / 'outputs/prefill-capacity-retest-20260914.json') == retest
    result = subprocess.run([sys.executable, '-I', str(root / 'VERIFY-RETEST.py')],
                            text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    replay = json.loads(result.stdout)
    assert replay['status'] == 'PASS_OFFLINE_RETEST_VERIFICATION'

readme = (REPO / 'README.md').read_text()
for cell in promotion['cold_cells']:
    assert f"{cell['aggregate_rate_from_mean_seconds']:,.0f} tok/s" in readme
repeat = {r['concurrency']: r for r in promotion['readme_runs']['benchmark-repeat']['rows']}
for value in [repeat[1]['coding_decode_per_stream']['mean'], repeat[8]['coding_full_batch_aggregate']['mean'],
              repeat[1]['prose_decode_aggregate']['mean'], repeat[8]['prose_decode_aggregate']['mean']]:
    assert f'{value:.2f} tok/s' in readme
assert '64K has not been measured' in readme
print(json.dumps(dict(status='PASS_PROMOTION_EVIDENCE', production_files=len(source['source_files']),
                      cold_cells=len(promotion['cold_cells']), readme_runs=len(promotion['readme_runs']),
                      retest_replay=replay, publication_audit=health['observed']), indent=2))
