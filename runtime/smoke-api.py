"""Run two bounded API smoke requests and preserve their inputs and outputs."""
import argparse
import json
import pathlib
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--base-url', default='http://127.0.0.1:8000')
parser.add_argument('--output', required=True)
args = parser.parse_args()
out = pathlib.Path(args.output)
out.mkdir(parents=True, exist_ok=True)

with urllib.request.urlopen(args.base_url + '/v1/models', timeout=10) as response:
    models = json.load(response)
assert 'deepseek-v41-flash' in [m['id'] for m in models['data']], models
(out / 'models.json').write_text(json.dumps(models, indent=2) + '\n')

context = '\n'.join(
    f'Archive entry {i}: This entry describes a routine inspection of shelves and labels.'
    for i in range(1, 81)
)
checks = [
    ('arithmetic', 'What is 17 times 19? Return only the integer.', '323', 32),
    ('long-context', context + '\nThe final inspection code is AMBER-742.\n'
     'What is the final inspection code? Return only the code.', 'AMBER-742', 32),
]
for name, prompt, expected, limit in checks:
    payload = {
        'model': 'deepseek-v41-flash',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0, 'max_tokens': limit,
        'chat_template_kwargs': {'thinking': False},
    }
    (out / f'{name}-request.json').write_text(json.dumps(payload, indent=2) + '\n')
    request = urllib.request.Request(
        args.base_url + '/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    elapsed = time.monotonic() - started
    content = result['choices'][0]['message'].get('content') or ''
    passed = content.strip() == expected
    record = {'elapsed_s': elapsed, 'expected': expected, 'passed': passed, 'response': result}
    (out / f'{name}-response.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'check': name, 'passed': passed, 'content': content,
                      'elapsed_s': elapsed, 'usage': result.get('usage')}), flush=True)
    assert passed, f'{name} produced unexpected content: {content!r}'
print('PASS: API model discovery, arithmetic, and longer chunked prefill.', flush=True)
