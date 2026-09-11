"""Launch the prepared native-resident SGLang comparison profile."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from configuration import engine_args, environment, load

p = argparse.ArgumentParser()
p.add_argument('--profile', default='/config/profile.json')
p.add_argument('--rank', type=int, required=True)
p.add_argument('--render', action='store_true')
a = p.parse_args()
c = load(a.profile)
args = engine_args(c, a.rank)
env = environment(c, a.rank)
receipt = dict(status='rendered' if a.render else 'launching_unvalidated_candidate', rank=a.rank,
               args=args, environment=env, profile=c)
if a.render:
    print(json.dumps(receipt, indent=2)); raise SystemExit(0)
lock = json.loads(Path('/opt/sglang8/versions.lock.json').read_text())
config = Path('/models') / c['model_subpath'] / 'config.json'
if hashlib.sha256(config.read_bytes()).hexdigest() != lock['model_config_sha256']:
    raise RuntimeError('Checkpoint config differs from the comparison reference')
if os.environ.get('SGLANG_BUILD_COMMIT') != lock['sglang_commit']:
    raise RuntimeError('SGLang image source revision differs from the prepared reference')
Path('/state').mkdir(exist_ok=True)
Path('/state/launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
os.environ.update(env)
os.execv(sys.executable, [sys.executable, '-m', 'sglang.launch_server', *args])
