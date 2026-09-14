"""Verify the public SG18 source snapshot and benchmark arithmetic on CPU."""
import hashlib
import json
from pathlib import Path
import statistics

here = Path(__file__).resolve().parent
repo = here.parents[2]
results = repo / 'sglang/results/sg18-indexer-mhc-wo'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, tolerance=1e-8):
    assert abs(actual - expected) <= tolerance, (actual, expected)


manifest = load(here / 'source-manifest.json')
assert len(manifest['production_files']) == 16
assert len(manifest['changed_from_sg17']) == 8
for name, digest in manifest['production_files'].items():
    assert sha(here / 'source' / name) == digest, name
for name, hashes in manifest['changed_from_sg17'].items():
    assert manifest['production_files'][name] == hashes['after_sha256'], name
assert sha(here / 'configuration.py') == manifest['configuration_sha256']
assert sha(here / 'sglang-indexer-mhc-wo.patch') == manifest['patch_sha256']
reference = load(results / 'reference-manifest.json')
assert len(reference['files']) == 35
for name, digest in reference['files'].items():
    assert sha(repo / name) == digest, name

comparison = load(results / 'comparison.json')
assert set(comparison['runs']) == {'sg17_initial', 'sg17_repeat', 'sg18_initial', 'sg18_repeat'}
coding_requests = prose_requests = 0
for name, run in comparison['runs'].items():
    folder = results / 'runs' / name
    coding = load(folder / 'coding.json')
    assert coding['source_sha256'] == reference['files']['bench/v41bench.py']
    expected_order = [(1, 1), (1, 4), (1, 8), (2, 8), (2, 4), (2, 1), (3, 1), (3, 4), (3, 8), (4, 1), (5, 1)]
    assert [(r['trial'], r['c']) for r in coding['batches']] == expected_order
    assert [r['c'] for r in coding['excluded_coding_warmups']] == [1, 4, 8]
    for batch in [*coding['excluded_coding_warmups'], *coding['batches']]:
        assert batch['tokens'] == 200 * batch['c']
        assert len(batch['requests']) == batch['c']
        for stream in batch['requests']:
            assert stream['prompt_tokens'] == 47 and stream['completion_tokens'] == 200
            close(stream['decode_tok_s'], 199 / (stream['total_s'] - stream['ttft_s']))
        close(batch['per_stream_tok_s'], round(statistics.mean(r['decode_tok_s'] for r in batch['requests']), 2))
        close(batch['ttft_mean_s'], round(statistics.mean(r['ttft_s'] for r in batch['requests']), 3))
        assert batch['wall_s'] + .000501 >= max(r['total_s'] for r in batch['requests'])
        assert batch['tokens'] / (batch['wall_s'] + .000501) - .00501 <= batch['agg_tok_s'] <= batch['tokens'] / (batch['wall_s'] - .000501) + .00501
    coding_requests += sum(r['c'] for r in coding['batches'])
    prose = load(folder / 'prose-metrics.json')
    assert prose['status'] == 'PASS' and len(prose['decode']) == len(prose['prefill']) == 3
    for trial in prose['decode']:
        assert trial['status'] == 'completed'
        assert [r['concurrency'] for r in trial['results']] == [1, 4, 8]
        for wave in trial['results']:
            c = wave['concurrency']
            assert not wave['error'] and wave['streamsFailed'] == 0
            assert wave['streamsOk'] == c == len(wave['streams'])
            assert wave['totalCompletionTokens'] == c * 256 and wave['totalDecodeTokens'] == c * 255
            for stream in wave['streams']:
                assert stream['completionTokens'] == 256 and stream['decodeTokens'] == 255
                assert stream['prefillTokens'] == (32 if c == 1 else 39)
                assert not stream['error'] and stream['finishReason'] == 'length'
                close(stream['decodeTps'], 255000 / stream['decodeMs'], .006)
            close(wave['meanDecodeTps'], statistics.mean(r['decodeTps'] for r in wave['streams']), .006)
            assert 0 < wave['aggregateDecodeTps'] <= sum(r['decodeTps'] for r in wave['streams']) + .02
            prose_requests += c
    for summary in run['rows']:
        c = summary['concurrency']
        batches = [r for r in coding['batches'] if r['c'] == c]
        waves = [r for t in prose['decode'] for r in t['results'] if r['concurrency'] == c]
        for key, values in [('coding_decode_per_stream', [r['per_stream_tok_s'] for r in batches]),
                            ('coding_full_batch_aggregate', [r['agg_tok_s'] for r in batches]),
                            ('coding_ttft_seconds', [r['ttft_mean_s'] for r in batches]),
                            ('prose_decode_aggregate', [r['aggregateDecodeTps'] for r in waves]),
                            ('prose_decode_per_stream', [r['meanDecodeTps'] for r in waves]),
                            ('prose_ttft_ms', [r['meanTtftMs'] for r in waves])]:
            assert values == summary[key]['values']
            close(summary[key]['mean'], statistics.mean(values))
            close(summary[key]['sample_stdev'], statistics.stdev(values))
            assert summary[key]['minimum'] == min(values) and summary[key]['maximum'] == max(values)
assert coding_requests == 164 and prose_requests == 156
for arm in ['sg17', 'sg18']:
    checks = load(results / arm / 'correctness.json')
    assert checks['before']['status'] == checks['after']['status'] == 'PASS'
    assert len(checks['arithmetic_c128']) == 128
    assert sorted(r['index'] for r in checks['arithmetic_c128']) == list(range(128))
    assert all(r['passed'] and r['actual'] == str(40 + r['index']) for r in checks['arithmetic_c128'])
    retrieval = load(results / arm / 'long-context.json')
    assert retrieval['status'] == 'PASS'
    assert [r['usage']['prompt_tokens'] for r in retrieval['cases']] == [32866, 131170, 299098]
    assert all(r['passed'] and r['actual'] == r['expected'] for r in retrieval['cases'])
dispatch = load(results / 'dispatch.json')
assert dispatch['status'] == 'PASS' and dispatch['phase_checks'] == len(dispatch['rows']) == 192
assert all(r['experts_identical'] for r in dispatch['rows'])
print(json.dumps(dict(status='PASS', source_files=16, unchanged_reference_files=35,
    measured_coding_requests=coding_requests, measured_prose_requests=prose_requests,
    limitation='Prose aggregate endpoint timestamps were not retained by the pinned runner; exact aggregate-window reconstruction remains unavailable.')))
