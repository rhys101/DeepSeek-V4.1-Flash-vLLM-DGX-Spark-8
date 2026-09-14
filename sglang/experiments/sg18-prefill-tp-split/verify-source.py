"""Verify the source snapshot and reconstruct both patches without a GPU."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'source-manifest.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for name, expected in manifest['packaged_files'].items():
    assert sha(ROOT / name) == expected, name
source = ROOT / 'source'
actual = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
assert actual == manifest['source_files'] and len(actual) == 18
inherited = {name.removeprefix('b12x/'): digest for name, digest in manifest['inherited_b12x_files'].items()}
sg17 = ROOT.parent / 'sg17-rocenante'
assert inherited == json.loads((sg17 / 'b12x-source-manifest.json').read_text())
with tarfile.open(sg17 / 'b12x-source.tar.gz') as archive:
    members = archive.getmembers()
    assert len(members) == len(inherited) == 241 and all(member.isfile() for member in members)
    assert {member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest() for member in members} == inherited
old = ROOT.parent / 'sg18-indexer-mhc-wo/source'
changes = manifest['changes_from_sg18']
changed = set()
for name, expected in actual.items():
    before = old / name
    if not before.exists() or sha(before) != expected:
        changed.add(name.removeprefix('python/'))
assert changed == set(changes)
with tempfile.TemporaryDirectory(prefix='prefill-patch-check-') as directory:
    work = Path(directory)
    shutil.copytree(old / 'python', work / 'python')
    shutil.copytree(ROOT / 'preimage/python', work / 'python', dirs_exist_ok=True)
    for name, hashes in changes.items():
        p = work / 'python' / name
        assert (sha(p) if p.exists() else None) == hashes['before_sha256'], name
    for patch in ['scratch-initialization.patch', 'native-prefill-tp.patch']:
        subprocess.run(['git', 'apply', '--check', str(ROOT / patch)], cwd=work / 'python', check=True)
        subprocess.run(['git', 'apply', str(ROOT / patch)], cwd=work / 'python', check=True)
    reconstructed = {str(p.relative_to(work)): sha(p) for p in work.rglob('*') if p.is_file()}
    assert reconstructed == actual

backend = 'python/sglang/srt/layers/attention/deepseek_v4_backend.py'
def functions(path):
    found = {}
    def visit(node, prefix=''):
        for child in ast.iter_child_nodes(node):
            qualified = prefix
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = prefix + child.name + '.'
                if not isinstance(child, ast.ClassDef):
                    found[qualified[:-1]] = child
            visit(child, qualified)
    visit(ast.parse(path.read_text()))
    return found
before, after = functions(old / backend), functions(source / backend)
assert before.keys() == after.keys()
changed_functions = {name for name in before if ast.dump(before[name]) != ast.dump(after[name])}
assert changed_functions == {'DeepseekV4AttnBackend.__init__', 'DeepseekV4AttnBackend._low_ratio_index_topk_dense'}, changed_functions
old_statements = before['DeepseekV4AttnBackend._low_ratio_index_topk_dense'].body
new_statements = after['DeepseekV4AttnBackend._low_ratio_index_topk_dense'].body
branch = next(node for node in new_statements if isinstance(node, ast.If) and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Attribute) and node.test.func.attr == 'eligible')
dump = lambda nodes: [ast.dump(node) for node in nodes]
fallback = dump(branch.orelse)
old_dump = dump(old_statements)
assert any(old_dump[i:i + len(fallback)] == fallback for i in range(len(old_dump)))

helper = source / 'python/sglang/srt/layers/attention/dsv4/spark_prefill_dense.py'
node = next(n for n in ast.parse(helper.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'partition')
namespace = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(helper), 'exec'), namespace)
partition = namespace['partition']
cases = 0
for rows in list(range(4097)) + [4097, 8191, 8192, 8193]:
    for world in [1, 2, 4, 8]:
        gathered, extents = [], []
        for rank in range(world):
            start, end, extent = partition(rows, world, rank)
            assert start % 128 == 0 and extent % 128 == 0
            owned = list(range(start, end))
            assert len(owned) <= extent
            gathered.extend(owned + [-1] * (extent - len(owned)))
            extents.append(extent)
        assert len(set(extents)) == 1
        assert gathered[:rows] == list(range(rows)) and all(x == -1 for x in gathered[rows:])
        cases += 1
print(json.dumps(dict(status='PASS_SOURCE_ONLY', production_files=len(actual), inherited_b12x_files=len(inherited), changed_files=len(changed), patches=2, partition_cases=cases, unchanged_fallback_ast=True), indent=2))
