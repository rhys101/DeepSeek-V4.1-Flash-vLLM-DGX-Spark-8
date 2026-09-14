"""Replay the original offline validators against byte-preserved public evidence.

Uses only the Python standard library. No network, model, GPU, or benchmark
client is invoked. Historical validator paths are recreated in a temporary
directory; the checkout and archives are never modified.
"""
import hashlib
from fractions import Fraction
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile

if sys.flags.optimize:
    raise RuntimeError('Run without -O: the historical validators use assertions.')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = REPO / 'sglang/results/sg18-prefill-tp-split'
MANIFEST = json.loads((RESULTS / 'evidence-manifest.json').read_text())


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproduce_recorded_stdev(recorded, replayed):
    """Reproduce Python 3.9's two-pass float variance and final square root.

    Newer statistics.stdev computes the square root of an exact fraction,
    changing some last bits. Verify both calculations from identical trial
    values, then use the recorded calculation for exact receipt comparison.
    No measurement or comparison tolerance is changed.
    """
    differences = 0
    assert recorded['runs'].keys() == replayed['runs'].keys()
    for phase, run in recorded['runs'].items():
        fresh = replayed['runs'][phase]
        assert len(run['rows']) == len(fresh['rows'])
        for original_row, fresh_row in zip(run['rows'], fresh['rows']):
            assert original_row.keys() == fresh_row.keys()
            for name, original in original_row.items():
                if not isinstance(original, dict) or 'sample_stdev' not in original:
                    continue
                current = fresh_row[name]
                values = original['values']
                assert values == current['values'] and len(values) > 1
                assert all(type(value) in (int, float) and math.isfinite(value) for value in values)
                center = statistics.mean(values)
                squares = sum((Fraction((value - center) ** 2) for value in values), Fraction())
                residual = sum((Fraction(value - center) for value in values), Fraction())
                variance = (squares - residual ** 2 / len(values)) / (len(values) - 1)
                legacy = math.sqrt(float(variance))
                assert original['sample_stdev'] == legacy, (phase, name, 'recorded stdev')
                assert current['sample_stdev'] == statistics.stdev(values), (phase, name, 'replayed stdev')
                differences += current['sample_stdev'] != legacy
                current['sample_stdev'] = legacy
    return differences


with tempfile.TemporaryDirectory(prefix='sg18-prefill-verify-') as folder:
    root = Path(folder)
    archived_files = 0
    for name, archive in MANIFEST['archives'].items():
        path = RESULTS / name
        assert digest(path) == archive['sha256'], name
        seen = set()
        with tarfile.open(path, 'r:gz') as tar:
            for member in tar:
                key = member.name
                relative = PurePosixPath(key)
                assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts, key
                assert key in archive['files'] and key not in seen, key
                seen.add(key)
                data = tar.extractfile(member).read()
                expected = archive['files'][key]
                assert len(data) == member.size == expected['bytes'], key
                assert hashlib.sha256(data).hexdigest() == expected['sha256'], key
                destination = root / key
                assert not destination.exists(), key
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
        assert seen == set(archive['files']), name
        archived_files += len(seen)

    pf = root / 'spark-prefill-tp-split-20260914'
    lab = root / 'spark-kernels-iter1'
    reference_path = root / MANIFEST['reference_manifest']
    reference = read(reference_path)
    assert len(reference['files']) == 35
    for name, sha in reference['files'].items():
        assert digest(REPO / name) == sha, name
        destination = reference_path.parent / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, destination)
    # The original README validator compares its prompts with these already
    # published SG17 reference records. It never invokes that repo's programs.
    for name in ['long-context.json', 'runs/sg17_repeat/prose-metrics.json']:
        relative = Path('sglang/results/sg17-rocenante') / name
        destination = root / 'readme-promotion-20260914/repo' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)

    replayed = []
    stdev_last_bit_differences = []

    def replay(script, args, destination, ignore=()):
        original_bytes = destination.read_bytes()
        original = json.loads(original_bytes)
        destination.unlink()
        result = subprocess.run([sys.executable, '-I', str(script), *args],
                                text=True, capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError(f'{script.name} {args} failed:\n{result.stdout}\n{result.stderr}')
        recomputed = read(destination)
        for key in ignore:
            original.pop(key, None)
            recomputed.pop(key, None)
        if script.name == 'validate-readme-suite.py':
            stdev_last_bit_differences.append(reproduce_recorded_stdev(original, recomputed))
        assert original == recomputed, (script.name, args)
        # Restore the recorded bytes, so downstream timestamp/hash checks use
        # the original receipts, after comparing every portable result above.
        destination.write_bytes(original_bytes)
        replayed.append(dict(script=script.name, args=args, status=original['status']))

    for label in MANIFEST['cold_runs']:
        replay(pf / 'validate-prefill-v01.py', [label], lab / label / 'prefill-validation.json')
    replay(pf / 'validate-prefill-partial-v01.py', [MANIFEST['partial_run']],
           lab / MANIFEST['partial_run'] / 'partial-validation.json')
    for label in MANIFEST['readme_runs']:
        # Only the absolute-path evidence dictionary varies with the temporary
        # directory. Source hashes, all measurements, answers, and guards match.
        replay(lab / 'validate-readme-suite.py', [label], lab / label / 'readme-validation.json',
               ignore=('evidence_sha256',))
    for label in [*MANIFEST['cold_runs'], *MANIFEST['readme_runs']]:
        destination = pf / (label + '-raw-guard-validation-v01.json')
        if destination.exists():
            replay(pf / 'validate-guard-samples-v01.py', [label], destination)
        else:
            result = subprocess.run([sys.executable, '-I', str(pf / 'validate-guard-samples-v01.py'), label],
                                    text=True, capture_output=True, timeout=120)
            assert result.returncode == 0, result.stderr
            assert read(destination)['status'] == 'PASS_RAW_GUARD_SAMPLES'
            replayed.append(dict(script='validate-guard-samples-v01.py', args=[label], status='PASS_RAW_GUARD_SAMPLES'))

    for candidate, controls in [
        ('prefill-native-v02', ['prefill-fixed-before-v01', 'prefill-fixed-after-v01']),
        ('prefill-native-confirmation-v01', ['prefill-fixed-after-v01', 'prefill-fixed-confirmation-after-v01'])
    ]:
        for count, kind in [(1, 'screen'), (2, 'bracket')]:
            replay(pf / 'compare-prefill-v01.py', [candidate, *controls[:count]],
                   pf / (candidate + '-timing-' + kind + '.json'))
        replay(pf / 'validate-temporal-bracket-v01.py', ['cold', candidate, *controls],
               pf / (candidate + '-temporal-bracket-v01.json'))
        readme_candidate = candidate.replace('prefill-', 'prefill-readme-', 1).replace('native-v02', 'native-v01')
        readme_controls = [name.replace('prefill-', 'prefill-readme-', 1) for name in controls]
        replay(pf / 'compare-readme-prefill-v01.py', [readme_candidate, *readme_controls],
               pf / (readme_candidate + '-comparison-v01.json'))
        replay(pf / 'validate-temporal-bracket-v01.py', ['readme', readme_candidate, *readme_controls],
               pf / (readme_candidate + '-temporal-bracket-v01.json'))

    capacity = MANIFEST['capacity_run']
    if capacity:
        replay(pf / 'validate-capacity-v01.py', [capacity], lab / capacity / 'capacity-validation.json')
        replay(pf / 'validate-guard-samples-v01.py', [capacity], pf / (capacity + '-raw-guard-validation-v01.json'))
    capacity_failure = MANIFEST.get('capacity_failure')
    if capacity_failure:
        assert not capacity
        replay(pf / 'validate-capacity-failure-v01.py', [], pf / 'prefill-capacity-failure-audit-v01.json')

    prerequisite = read(pf / 'prefill-confirmation-decision-v01.json')
    assert prerequisite['status'] == 'PASS_PRECAPACITY'
    for name, sha in prerequisite['evidence_sha256'].items():
        assert digest(pf / name) == sha, name
    qualified = read(lab / 'components-qualified-prefillnative01.json')
    assert qualified['status'] == 'PASS' and qualified['trial'] == 'nativecomp03'
    for name, sha in qualified['receipt']['tests'].items():
        assert digest(lab / 'components-nativecomp03/tests' / name) == sha, name
    manifest = read(HERE / 'source-manifest.json')
    for name, sha in manifest['source_files'].items():
        assert digest(HERE / 'source' / name) == sha
        assert qualified['receipt']['candidate_files']['production/' + name] == sha, name

    # Cross-check the convenient JSON export against the original receipts.
    summary = read(RESULTS / 'measurements.json')
    for label, block in summary['cold_runs'].items():
        assert block['validation'] == read(lab / label / 'prefill-validation.json')
        raw = read(lab / label / 'results.json')['rows']
        assert len(block['all_batches']) == len(raw) == 24
        for exported, recorded in zip(block['all_batches'], raw):
            assert all(recorded[key] == value for key, value in exported.items())
    for label, block in summary['readme_runs'].items():
        assert block['runs'] == read(lab / label / 'readme-validation.json')['runs']
    for name, comparison in summary['comparisons'].items():
        assert comparison == read(pf / name), name
    assert bool(summary['capacity_run']) == bool(capacity)
    assert bool(summary.get('capacity_failure')) == bool(capacity_failure)
    if capacity_failure:
        assert summary['capacity_failure'] == read(pf / 'prefill-capacity-failure-audit-v01.json')
    if summary.get('final_decision'):
        decision = read(pf / 'prefill-final-decision-v01.json')
        assert summary['final_decision'] == decision
        assert summary['status'] == MANIFEST['status'] == decision['status']
        assert decision['handoff_audit_status'] == 'PASS'
        for name, sha in decision['evidence_sha256'].items():
            assert digest(root / name) == sha, name
        if capacity_failure:
            assert decision['status'] == 'NOT_PROMOTED_BASELINE_RESTORED'
            assert decision['active_service'] == 'scratchfix01'
            assert read(lab / 'prefill-final-baseline-audit-v01.json')['status'] == 'PASS'

    print(json.dumps(dict(status='PASS_RECORDED_EVIDENCE', archived_files=archived_files,
                          cold_blocks=len(MANIFEST['cold_runs']), readme_blocks=len(MANIFEST['readme_runs']),
                          completed_cold_batches=24 * len(MANIFEST['cold_runs']),
                          interrupted_batches=13, validations_replayed=len(replayed),
                          unchanged_reference_files=35, capacity_validation_included=bool(capacity),
                          capacity_failure_verified=bool(capacity_failure),
                          recorded_stdev_method='Python 3.9 two-pass float variance, then sqrt',
                          replay_stdev_last_bit_differences=sum(stdev_last_bit_differences),
                          initial_timing_status=read(pf / 'prefill-native-v02-timing-bracket.json')['status'],
                          confirmation_timing_status=read(pf / 'prefill-native-confirmation-v01-timing-bracket.json')['status'],
                          limitation='Recomputes archived records. Does not repeat GPU execution or establish broad model-quality parity.'), indent=2))
