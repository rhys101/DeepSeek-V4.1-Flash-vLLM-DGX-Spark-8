#!/usr/bin/env python3
"""Operate the EP4 snapshot from rank zero over the configured fabric."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'runtime'))
from configuration import engine_args, environment, load as load_runtime

def load(path):
    c = load_runtime(path)
    fixed = dict(ep_size=4, min_free_slots_delay=1, retry_free_admission=False,
                 chunked_prefill_size=8192, max_prefill_tokens=8192,
                 mem_fraction_static=0.80, max_total_tokens=3200000,
                 nccl_channels=8, nccl_algo='')
    for key, value in fixed.items():
        if c.get(key, value if key in ('nccl_channels', 'nccl_algo') else None) != value:
            raise ValueError(f'The published EP4 profile requires {key}={value!r}')
    if c.get('minimum_available_gib', 0) < 13:
        raise ValueError('Keep the validated 13 GiB minimum OS reserve')
    if c['run_dir'] == '/' or any(',' in c[k] for k in ('run_dir', 'model_store')):
        raise ValueError('Use a dedicated deployment directory and comma-free mount paths')
    return c

def container_name(rank):
    return f'sglang8-ep4-r{rank}'

def call(c, n, command, **kwargs):
    argv = ['bash', '-c', command] if n['rank'] == 0 else [
        'ssh', '-b', c['head_ip'], '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        n['host'].split('@')[0] + '@' + n['ip'], command]
    return subprocess.run(argv, check=True, **kwargs)

def read(c, n, command):
    return call(c, n, command, capture_output=True, text=True, timeout=120).stdout

def require_head(c):
    data = json.loads(subprocess.check_output(['ip', '-j', 'addr']))
    ips = {a['local'] for interface in data for a in interface.get('addr_info', [])}
    if c['head_ip'] not in ips: raise RuntimeError('Run cluster operations from Spark1')
    for n in c['workers'][1:]:
        route = json.loads(subprocess.check_output(['ip', '-j', 'route', 'get', n['ip'], 'from', c['head_ip']]))[0]
        if route['dev'] != c['fabric_interface']: raise RuntimeError('Peer route leaves the configured fabric')

def expected(c):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', c.get('expected_image_id') or ''):
        raise RuntimeError('Build the image first; expected_image_id must identify the actual local build')

def image_check(c, n):
    actual = read(c, n, shlex.join(['docker', 'image', 'inspect', '--format', '{{.Id}}', c['image']])).strip()
    if actual != c['expected_image_id']: raise RuntimeError(f"Image mismatch on rank {n['rank']}")

def idle(c, n):
    names = read(c, n, 'docker ps -q').splitlines()
    if names:
        data = json.loads(read(c, n, shlex.join(['docker', 'inspect', *names])))
        if any(d['HostConfig'].get('DeviceRequests') for d in data):
            raise RuntimeError(f"A GPU container is already running on rank {n['rank']}")
    pids = read(c, n, 'nvidia-smi --query-compute-apps=pid --format=csv,noheader').strip()
    if pids: raise RuntimeError(f"GPU processes present on rank {n['rank']}: {pids}")

def command(c, rank, action='serve'):
    name = container_name(rank)
    d = c['run_dir']
    cmd = ['docker', 'run', '--pull', 'never']
    cmd += ['-d', '--name', name] if action == 'serve' else ['--rm', '--name', f'sglang8-ep4-check-r{rank}']
    cmd += ['--gpus', 'all', '--network', 'host', '--ipc', 'host', '--device', '/dev/infiniband',
        '--cap-add', 'IPC_LOCK', '--ulimit', 'memlock=-1', '--ulimit', 'nofile=1048576:1048576',
        '--mount', f"type=bind,src={c['model_store']},dst=/models,readonly",
        '--mount', f'type=bind,src={d}/kit/configs/cluster.json,dst=/config/profile.json,readonly',
        '--mount', f'type=bind,src={d}/cache,dst=/cache',
        '--mount', f'type=bind,src={d}/state,dst=/state']
    for key, value in environment(c, rank).items(): cmd += ['-e', key + '=' + value]
    if action == 'nccl':
        cmd += ['--entrypoint', 'torchrun', c['image'], '--nnodes', '8', '--nproc-per-node', '1',
                '--node-rank', str(rank), '--master-addr', c['head_ip'], '--master-port', str(c['nccl_test_port']),
                '--rdzv-conf', 'timeout=180', '/opt/sglang8/runtime/check-fabric.py']
    elif action == 'preflight':
        cmd += ['--entrypoint', 'python3', c['image'], '/opt/sglang8/runtime/check-resident.py']
    elif action == 'dense':
        cmd += ['--mount', f'type=bind,src={d}/kit/validation,dst=/validation,readonly',
                '--entrypoint', 'python3', c['image'], '/validation/check-mxfp8.py']
    else:
        cmd += [c['image'], '--rank', str(rank)]
    return cmd

def mapped(c, fn):
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(fn, c['workers']))

def stage(c):
    # A staged profile is immutable. Choose a fresh run_dir for a new attempt.
    mapped(c, lambda n: call(c, n, shlex.join(['test', '!', '-e', c['run_dir'] + '/kit'])))
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp)/'kit.tar'
        with tarfile.open(archive, 'w') as tar:
            for folder in ('adapter','runtime','scripts','docker','validation','bench'):
                tar.add(ROOT/folder, arcname=folder, filter=lambda x: None if '__pycache__' in x.name else x)
            for name in ('versions.lock.json','LICENSE','LICENSE.sglang','LICENSE.upstream-MIT','LICENSE.benchmark-MIT','NOTICE','README.md'):
                tar.add(ROOT/name, arcname=name)
            config = Path(tmp)/'cluster.json'; config.write_text(json.dumps(c, indent=2)+'\n')
            tar.add(config, arcname='configs/cluster.json')
        for n in c['workers']:
            run_dir = c['run_dir']
            call(c, n, shlex.join(['mkdir','-p',run_dir+'/kit',run_dir+'/cache',run_dir+'/state']))
            with archive.open('rb') as source:
                call(c, n, shlex.join(['tar','-xf','-','-C',run_dir+'/kit']), stdin=source, timeout=120)
            print(f"Staged isolated SGLang kit on rank {n['rank']}", flush=True)

def rank_status(c, n):
    data = json.loads(read(c, n, shlex.join(['docker', 'inspect', container_name(n['rank'])])))[0]
    if data['Image'] != c['expected_image_id']:
        raise RuntimeError(f"Container image mismatch on rank {n['rank']}")
    memory = read(c, n, 'cat /proc/meminfo')
    available = int(next(line.split()[1] for line in memory.splitlines() if line.startswith('MemAvailable:'))) / 1024**2
    return dict(rank=n['rank'], state=data['State'], restarts=data['RestartCount'], available_gib=available)

def validate_layout_record(layout, rank):
    for role, count, experts in [('target',40,96), ('draft',3,32)]:
        if len(layout[role]) != count:
            raise RuntimeError(f'Unexpected {role} module count on rank {rank}')
        expected = dict(method='Mxfp4FlashinferCutlassMoEMethod',local_experts=experts,
                        intermediate=1152,ep_size=4,ep_rank=rank//2,tp_size=2,tp_rank=rank%2,
                        w13_shape=[experts,2304,2560],w2_shape=[experts,5120,576])
        for name, actual in layout[role].items():
            if actual != expected:
                raise RuntimeError(f'Unexpected expert layout on rank {rank}, {name}: {actual}')

def selected_layout(c, n):
    result = call(c, n, shlex.join(['docker','logs',container_name(n['rank'])]), capture_output=True, text=True, timeout=60)
    records = [json.loads(line.split('DSPARK_MOE_LAYOUT ',1)[1])
               for line in (result.stdout+'\n'+result.stderr).splitlines() if 'DSPARK_MOE_LAYOUT ' in line]
    if not records:
        raise RuntimeError(f"No actual expert-layout receipt on rank {n['rank']}")
    # Docker preserves older logs when a retained container restarts.
    validate_layout_record(records[-1], n['rank'])
    return dict(rank=n['rank'],layout=records[-1])

def wait_ready(c):
    deadline = time.monotonic() + 2400
    records = Path(c['run_dir']) / 'state'
    while time.monotonic() < deadline:
        ranks = mapped(c, lambda n: rank_status(c, n))
        (records / 'startup-ranks.json').write_text(json.dumps(ranks, indent=2) + '\n')
        if any(not r['state']['Running'] or r['state']['OOMKilled'] or r['restarts'] for r in ranks):
            raise RuntimeError('A serving rank exited, restarted or was OOM-killed; retained logs identify it')
        minimum = min(r['available_gib'] for r in ranks)
        if minimum < c['minimum_available_gib']:
            raise RuntimeError(f'OS reserve breached: {minimum:.2f} GiB')
        try:
            # The configured bind address may be a particular interface.
            host = '127.0.0.1' if c['api_host'] in ('0.0.0.0', '127.0.0.1') else c['api_host']
            with urllib.request.urlopen(f"http://{host}:{c['api_port']}/health", timeout=4) as response:
                healthy = response.status == 200
        except (OSError, TimeoutError):
            healthy = False
        if healthy:
            with urllib.request.urlopen(f"http://{host}:{c['api_port']}/get_server_info", timeout=20) as response:
                info = json.load(response)
            for key, expected_value in dict(tp_size=8, ep_size=4, speculative_dspark_block_size=5, min_free_slots_delay=1, moe_runner_backend='flashinfer_mxfp4').items():
                if info[key] != expected_value:
                    raise RuntimeError(f'Unexpected server setting {key}: {info[key]}')
            layouts = mapped(c, lambda n: selected_layout(c, n))
            (records / 'selected-moe-layout.json').write_text(json.dumps(layouts, indent=2) + '\n')
            (records / 'server-info.json').write_text(json.dumps(info, indent=2) + '\n')
            print(f'EP4 ready on eight ranks; minimum OS-available memory {minimum:.2f} GiB')
            return
        print(f'Loading; minimum OS-available memory {minimum:.2f} GiB', flush=True)
        time.sleep(20)
    raise TimeoutError('EP4 did not become ready within 40 minutes')

def distribute(c):
    expected(c); image_check(c, c['workers'][0])
    archive = Path(c['run_dir'])/'sglang8-image.tar'; archive.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['docker','save','-o',str(archive),c['image']],check=True)
    with archive.open('rb') as source: digest = hashlib.file_digest(source, 'sha256').hexdigest()
    def transfer(n):
        remote = str(archive)
        call(c,n,shlex.join(['mkdir','-p',c['run_dir']]))
        ssh = shlex.join(['ssh','-b',c['head_ip'],'-o','BatchMode=yes','-o','ConnectTimeout=15'])
        peer = n['host'].split('@')[0]+'@'+n['ip']
        subprocess.run(['rsync','-a','--partial','--protect-args','-e',ssh,str(archive),peer+':'+remote],check=True)
        if read(c,n,shlex.join(['sha256sum',remote])).split()[0] != digest: raise RuntimeError('Image archive checksum mismatch')
        call(c,n,shlex.join(['docker','load','-i',remote]),timeout=900)
        image_check(c,n)
        print(f"Verified image on rank {n['rank']}",flush=True)
    # The model is stopped during distribution. Three independent workers can
    # copy/load concurrently while retaining per-worker hash and image checks.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(transfer, c['workers'][1:]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['dry-run','stage','distribute','preflight','dense','nccl','serve','start','status','stop'])
    p.add_argument('--config', default=str(ROOT/'configs/cluster.local.json'))
    a=p.parse_args(); c=load(a.config)
    if a.action=='dry-run':
        print(json.dumps([dict(rank=n['rank'], docker_command=command(c,n['rank']), engine_args=engine_args(c,n['rank'])) for n in c['workers']],indent=2));return
    require_head(c)
    if a.action=='stage': stage(c);return
    if a.action=='distribute': distribute(c);return
    expected(c)
    if a.action in ('serve','start','preflight','dense','nccl'):
        mapped(c, lambda n:image_check(c,n)); mapped(c,lambda n:idle(c,n))
        if a.action=='dense':
            call(c,c['workers'][0],shlex.join(command(c,0,'dense')),timeout=1200)
            return
        if a.action=='preflight':
            def check_rank(n):
                try:
                    r=call(c,n,shlex.join(command(c,n['rank'],'preflight')),timeout=600,capture_output=True,text=True)
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(f"Resident check rank {n['rank']} failed: {exc.stdout[-3000:]} {exc.stderr[-3000:]}") from exc
                result=dict(rank=n['rank'],image=c['expected_image_id'],stdout=r.stdout,stderr=r.stderr)
                (Path(c['run_dir'])/'state'/f"resident-check-r{n['rank']}.json").write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps(dict(rank=n['rank'],status='PASS',output=r.stdout)),flush=True)
            mapped(c,check_rank)
            return
        if a.action=='nccl':
            mapped(c,lambda n:call(c,n,shlex.join(command(c,n['rank'],'nccl')),timeout=600))
            return
        if a.action=='start':
            for n in c['workers']:
                d=json.loads(read(c,n,shlex.join(['docker','inspect',container_name(n['rank'])])))[0]
                if d['Image']!=c['expected_image_id']: raise RuntimeError('Existing container image mismatch')
        started=[]
        try:
            for n in c['workers'][1:]+c['workers'][:1]:
                cmd=command(c,n['rank']) if a.action=='serve' else ['docker','start',container_name(n['rank'])]
                call(c,n,shlex.join(cmd),timeout=120);started.append(n)
            wait_ready(c)
        except Exception:
            mapped(c, lambda n: call(c,n,shlex.join(['docker','stop','-t','60',container_name(n['rank'])]),timeout=120) if n in started else None)
            raise
        print('Run capability checks before benchmarking. Containers and logs are retained.');return
    for n in c['workers']:
        name=container_name(n['rank'])
        d=json.loads(read(c,n,shlex.join(['docker','inspect',name])))[0]
        if d['Image']!=c['expected_image_id']: raise RuntimeError('Container image differs from configured candidate')
        if a.action=='stop': call(c,n,shlex.join(['docker','stop','-t','60',name]),timeout=120)
        else:
            mem=read(c,n,"awk '/^MemAvailable:|^SwapTotal:|^SwapFree:/ {print}' /proc/meminfo")
            available=int(next(line.split()[1] for line in mem.splitlines() if line.startswith('MemAvailable:')))/1024**2
            print(json.dumps(dict(rank=n['rank'],state=d['State'],restarts=d['RestartCount'],memory=mem,
                available_gib=available,meets_headroom=available>=c['minimum_available_gib'])))

if __name__=='__main__': main()
