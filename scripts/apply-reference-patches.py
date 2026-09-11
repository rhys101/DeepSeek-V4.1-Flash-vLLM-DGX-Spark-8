"""Apply a checked source patch to an exact feature-branch tree."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import vllm

manifest_path=Path(sys.argv[1]).resolve()
manifest=json.loads(manifest_path.read_text())
root=Path(vllm.__file__).resolve().parent

def verify(expected):
    for name,digest in expected.items():
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual!=digest:raise RuntimeError(f'Unexpected source bytes: {name}')

verify(manifest['original_files'])
subprocess.run(['patch','--batch','--forward','--fuzz=0','-p1','-d',str(root.parent),
                '-i',str(manifest_path.parent/manifest['patch'])],check=True)
verify(manifest['result_files'])
print(f"Verified {manifest['patch']}: {len(manifest['result_files'])} source hashes match")
