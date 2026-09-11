#!/usr/bin/env python3
"""Configure and operate eight Sparks from rank 0 over a source-bound fabric."""
import argparse
import concurrent.futures
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]


def load_config(path):
    c = json.loads(Path(path).read_text())
    if c.get('schema_version') != 1:
        raise ValueError('Expected schema_version 1')
    nodes = c['nodes']
    if len(nodes) != 8 or sorted(n['rank'] for n in nodes) != list(range(8)):
        raise ValueError('Supply exactly eight nodes with unique ranks 0..7')
    c['nodes'] = sorted(nodes, key=lambda n: n['rank'])
    addresses = []
    for n in nodes:
        addresses.append(str(ipaddress.IPv4Address(n['fabric_ip'])))
        for key in ('ssh_user', 'fabric_interface'):
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', n[key]):
                raise ValueError(f'Invalid {key}')
        if not re.fullmatch(r'[=A-Za-z0-9_:,.-]+', n['nccl_hcas']):
            raise ValueError('Invalid NCCL HCA selection')
    if len(set(addresses)) != 8:
        raise ValueError('Fabric addresses must be unique')
    for key in ('model_store', 'deployment_dir'):
        p = Path(c[key])
        if not p.is_absolute() or '..' in p.parts or any(x in str(p) for x in '\n\r\x00'):
            raise ValueError(f'{key} must be an absolute path without traversal')
    p = Path(c['model_subpath'])
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('model_subpath must be inside model_store')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]*', c['container_prefix']):
        raise ValueError('Invalid container_prefix')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/:@-]*', c['image']):
        raise ValueError('Invalid image reference')
    ipaddress.ip_address(c['api_host'])
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', c['expected_image_id']):
        raise ValueError('Set expected_image_id to the local result of docker image inspect')
    if not isinstance(c['attempt'], int) or c['attempt'] < 1:
        raise ValueError('attempt must be a positive integer')
    for key in ('api_port', 'master_port', 'nccl_test_port'):
        if not isinstance(c[key], int) or not 1024 <= c[key] <= 65535:
            raise ValueError(f'Invalid {key}')
    profile_path = (Path(path).resolve().parent / c['profile']).resolve()
    p = json.loads(profile_path.read_text())
    for key in ('max_num_seqs', 'max_model_len', 'max_num_batched_tokens', 'spec_k'):
        if type(p[key]) is not int:
            raise ValueError(f'{key} must be an integer')
    for key in ('graphs', 'prefix_cache', 'vision'):
        if type(p[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    if p['spec_method'] not in ('none', 'dspark') or p['spec_k'] != 5:
        raise ValueError('This release supports native DSpark k=5 or the eager baseline')
    if not 1 <= p['max_num_seqs'] <= 8 or not 8192 <= p['max_model_len'] <= 300000:
        raise ValueError('Profile exceeds the documented launch range')
    if not 0.5 <= p['gpu_memory_utilization'] <= 0.80:
        raise ValueError('Profile exceeds the documented memory range')
    if not 1 <= p['max_num_batched_tokens'] <= 8192:
        raise ValueError('Batches must be between 1 and 8192 tokens')
    runtime_env = p.get('runtime_env', {})
    supported_env = {'NCCL_CUMEM_ENABLE', 'NCCL_NVLS_ENABLE', 'PYTORCH_CUDA_ALLOC_CONF',
                     'VLLM_USE_FLASHINFER_SAMPLER', 'CUDA_LOG_FILE', 'NCCL_BUFFSIZE',
                     'NCCL_LL128_BUFFSIZE', 'NCCL_PROTO', 'NCCL_MAX_NCHANNELS',
                     'SPARK_MXFP8_B12X_MAX_M'}
    if not isinstance(runtime_env, dict) or set(runtime_env) - supported_env:
        raise ValueError('Unsupported runtime environment setting')
    if any(not isinstance(v, str) or any(x in v for x in '\n\r\x00')
           for v in runtime_env.values()):
        raise ValueError('Runtime environment values must be single-line strings')
    c['settings'] = p
    return c


def env_text(c, n):
    p = c['settings']
    values = dict(IMAGE=c['image'], EXPECTED_IMAGE_ID=c['expected_image_id'],
                  NODE_RANK=n['rank'], HEAD_IP=c['nodes'][0]['fabric_ip'],
                  NODE_IP=n['fabric_ip'], FABRIC_IF=n['fabric_interface'], NCCL_HCAS=n['nccl_hcas'],
                  MODEL_STORE=c['model_store'], MODEL_SUBPATH=c['model_subpath'], RUN_DIR=c['deployment_dir'],
                  API_HOST=c['api_host'], API_PORT=c['api_port'], MASTER_PORT=c['master_port'],
                  NCCL_TEST_PORT=c['nccl_test_port'], MAX_MODEL_LEN=p['max_model_len'],
                  MAX_BATCH_TOKENS=p['max_num_batched_tokens'], MAX_NUM_SEQS=p['max_num_seqs'],
                  GPU_MEMORY_UTILIZATION=p['gpu_memory_utilization'], ENGRAM_CPU_OFFLOAD='false',
                  ENABLE_VISION=int(p['vision']), ENABLE_GRAPHS=int(p['graphs']), ENABLE_PREFIX_CACHE=int(p['prefix_cache']),
                  RUST_FRONTEND=0, ATTEMPT=c['attempt'], SPEC_METHOD=p['spec_method'], SPEC_K=p['spec_k'],
                  KV_CACHE_DTYPE=p['kv_cache_dtype'], CONTAINER_PREFIX=c['container_prefix'])
    values.update(p.get('runtime_env', {}))
    return ''.join(f'{key}={shlex.quote(str(value))}\n' for key, value in values.items())


def node_command(c, n, command):
    if n['rank'] == 0:
        return ['bash', '-c', command]
    return ['ssh', '-b', c['nodes'][0]['fabric_ip'], '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            f"{n['ssh_user']}@{n['fabric_ip']}", command]


def run(c, n, command, **kwargs):
    return subprocess.run(node_command(c, n, command), check=True, **kwargs)


def require_head(c):
    data = json.loads(subprocess.check_output(['ip', '-j', 'addr']))
    local = {a['local'] for interface in data for a in interface.get('addr_info', [])}
    if c['nodes'][0]['fabric_ip'] not in local:
        raise ValueError('Run cluster operations on rank 0; its configured fabric IP is not local')
    for n in c['nodes'][1:]:
        route = json.loads(subprocess.check_output(['ip', '-j', 'route', 'get', n['fabric_ip'],
                                                   'from', c['nodes'][0]['fabric_ip']]))[0]
        if route['dev'] != c['nodes'][0]['fabric_interface']:
            raise ValueError(f"Route to rank {n['rank']} leaves the configured fabric")


def runner(c, action):
    d = c['deployment_dir']
    return shlex.join(['bash', f'{d}/kit/run-node.sh', action, f'{d}/node.env'])


def api_base(c):
    address = ipaddress.ip_address(c['api_host'])
    if address.is_unspecified:
        address = ipaddress.ip_address('::1' if address.version == 6 else '127.0.0.1')
    host = f'[{address}]' if address.version == 6 else str(address)
    return f"http://{host}:{c['api_port']}"


def configure(c):
    for n in c['nodes']:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path/'node.env').write_text(env_text(c, n))
            os.chmod(path/'node.env', 0o600)
            with tarfile.open(path/'kit.tar', 'w') as tar:
                tar.add(path/'node.env', arcname='node.env')
                tar.add(REPO/'runtime', arcname='kit', filter=lambda x: None if '__pycache__' in x.name else x)
            d = shlex.quote(c['deployment_dir'])
            # Existing configuration requires an explicit new deployment directory.
            command = f'mkdir -p {d} && test ! -e {d}/node.env && tar -xf - -C {d}'
            with (path/'kit.tar').open('rb') as archive:
                run(c, n, command, stdin=archive)
        print(f"Configured rank {n['rank']}", flush=True)


def launch(c):
    for n in c['nodes']:
        running = run(c, n, shlex.join(['docker', 'ps', '--filter',
                       f"name={c['container_prefix']}-a", '--format', '{{.Names}}']),
                       capture_output=True, text=True).stdout.strip()
        if running:
            raise ValueError(f"Stop the previous deployment before launching: {running}")
        name = f"{c['container_prefix']}-a{c['attempt']}-r{n['rank']}"
        check = subprocess.run(node_command(c, n, shlex.join(['docker', 'container', 'inspect', name])),
                               capture_output=True)
        if check.returncode == 0:
            raise ValueError(f'{name} already exists; select a new attempt and deployment directory')
        run(c, n, runner(c, 'dry-run'), stdout=subprocess.DEVNULL)
    try:
        import socket
        from urllib.parse import urlsplit
        connection = socket.create_connection((urlsplit(api_base(c)).hostname, c['api_port']), timeout=1)
    except OSError:
        pass
    else:
        connection.close()
        raise ValueError('The requested API port is already in use')
    for n in c['nodes'][1:] + c['nodes'][:1]:
        run(c, n, runner(c, 'serve'))
    deadline = time.monotonic() + 1800
    endpoint = api_base(c)
    while time.monotonic() < deadline:
        try:
            if urllib.request.urlopen(endpoint+'/health', timeout=5).status == 200:
                print('API ready. Run the inference smoke test before benchmarking.')
                return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(10)
    raise TimeoutError('API did not become ready in 30 minutes; containers and logs retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'configure', 'preflight', 'validate-config',
                                         'dry-run', 'launch', 'status', 'stop', 'nccl', 'smoke'])
    parser.add_argument('config')
    parser.add_argument('--nodes', type=int, choices=[2, 4, 8], default=8, help='NCCL test ranks')
    a = parser.parse_args()
    c = load_config(a.config)
    if a.action == 'validate':
        print('Configuration valid; no host operations performed.')
        return
    require_head(c)
    lock_path = Path(c['deployment_dir'])
    lock_path.mkdir(parents=True, exist_ok=True)
    with (lock_path/'.controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.action == 'configure':
            configure(c)
        elif a.action == 'launch':
            launch(c)
        elif a.action == 'smoke':
            subprocess.run([sys.executable, str(REPO/'runtime/smoke-api.py'), '--base-url',
                            api_base(c), '--output', str(lock_path/'smoke')], check=True)
        elif a.action == 'nccl':
            def one(n):
                return run(c, n, f'NCCL_TEST_NODES={a.nodes} '+runner(c, 'nccl'))
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.nodes) as pool:
                list(pool.map(one, c['nodes'][:a.nodes]))
        else:
            for n in c['nodes']:
                print(f"Rank {n['rank']}: {a.action}", flush=True)
                run(c, n, runner(c, a.action))


if __name__ == '__main__':
    main()
