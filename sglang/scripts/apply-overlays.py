"""Verify every original/source hash before installing the eight EP4 files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def apply(source, target):
    manifest = json.loads((source / 'manifest.json').read_text())
    checked = []
    for name, hashes in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe overlay path: {name}')
        original, replacement = target / relative, source / 'source' / relative
        if hashes['before_sha256'] is None:
            if original.exists():
                raise ValueError(f'New source already exists: {name}')
        elif hashlib.sha256(original.read_bytes()).hexdigest() != hashes['before_sha256']:
            raise ValueError(f'Base source differs: {name}')
        if hashlib.sha256(replacement.read_bytes()).hexdigest() != hashes['after_sha256']:
            raise ValueError(f'Overlay source differs: {name}')
        checked.append((replacement, original, hashes['after_sha256']))
    for replacement, original, expected in checked:
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(replacement, original)
        assert hashlib.sha256(original.read_bytes()).hexdigest() == expected
    return len(checked)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    print(f'EP4_SOURCE_HASHES_PASS: {apply(args.source, args.target)} files')
