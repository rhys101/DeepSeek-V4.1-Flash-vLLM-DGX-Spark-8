#!/usr/bin/env python3
"""Copy an image or checkpoint from rank 0 over the configured fabric."""
import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

from cluster import load_config, require_head, run

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('kind', choices=['image', 'model'])
p.add_argument('config')
a = p.parse_args()
c = load_config(a.config)
require_head(c)
root = Path(c['deployment_dir'])
root.mkdir(parents=True, exist_ok=True)
if a.kind == 'image':
    actual = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', c['image']], text=True).strip()
    if actual != c['expected_image_id']:
        raise SystemExit('Head image ID differs from the configured build')
    archive = root/'image.tar'
    subprocess.run(['docker', 'save', '-o', str(archive), c['image']], check=True)
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    expected = h.hexdigest()
    (root/'image.sha256').write_text(expected+'  image.tar\n')
for n in c['nodes'][1:]:
    ssh = ['ssh', '-b', c['nodes'][0]['fabric_ip'], '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
    peer = f"{n['ssh_user']}@{n['fabric_ip']}"
    if a.kind == 'image':
        run(c, n, shlex.join(['mkdir', '-p', c['deployment_dir']]))
        subprocess.run(['rsync', '-a', '--partial', '--protect-args', '--info=progress2', '-e', shlex.join(ssh),
                        str(archive), f"{peer}:{c['deployment_dir']}/image.tar"], check=True)
        remote = c['deployment_dir']+'/image.tar'
        received = run(c, n, shlex.join(['sha256sum', remote]), capture_output=True, text=True).stdout.split()[0]
        if received != expected:
            raise SystemExit(f"Image checksum mismatch on rank {n['rank']}")
        run(c, n, shlex.join(['docker', 'load', '-i', remote]))
        actual = run(c, n, shlex.join(['docker', 'image', 'inspect', '--format', '{{.Id}}', c['image']]),
                     capture_output=True, text=True).stdout.strip()
        if actual != c['expected_image_id']:
            raise SystemExit(f"Imported image ID mismatch on rank {n['rank']}")
    else:
        model = str(Path(c['model_store'])/c['model_subpath'])
        if not (Path(model)/'model.safetensors.index.json').is_file():
            raise SystemExit('Checkpoint index is missing on rank 0')
        run(c, n, shlex.join(['mkdir', '-p', model]))
        subprocess.run(['rsync', '-aL', '--checksum', '--partial', '--protect-args', '--info=progress2',
                        '-e', shlex.join(ssh), model+'/', f'{peer}:{model}/'], check=True)
    print(f"Copied {a.kind} to rank {n['rank']}", flush=True)
