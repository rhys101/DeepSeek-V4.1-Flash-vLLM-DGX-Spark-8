"""Exercise source-extracted context methods without GPU imports or model weights.

The real source functions run against minimal flag/group metadata. Full model
selection is checked separately from every rank's startup method receipt.
"""
import ast
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

p=Path(__file__).resolve().parent
source=p.parent/'patches/source/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py'
original=p/'reference/dspark_worker_v2.py'
utils=p/'reference/moe_utils.py'
flags=SimpleNamespace(moe=SimpleNamespace(runner_backend='target-b12x'))
draft_backend='draft-flashinfer'
events=[]
@contextmanager
def draft_tp_context(group):
    events.append(('enter_dp',group))
    try: yield
    finally: events.append(('exit_dp',group))

namespace=dict(contextmanager=contextmanager,nullcontext=nullcontext,
    draft_tp_context=draft_tp_context,get_flags=lambda:flags,
    get_speculative_moe_runner_backend=lambda:draft_backend,
    get_parallel=lambda:SimpleNamespace(attn_tp_group='draft-group'))
tree=ast.parse(utils.read_text())
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='speculative_moe_backend_context')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(utils),'exec'),namespace)
def extract(path):
    tree=ast.parse(path.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='DSparkWorkerV2')
    fn=copy.deepcopy(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_draft_context'))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['_draft_context'],tree

old,_=extract(original)
with old(SimpleNamespace(_draft_dp_context_enabled=False)):
    assert flags.moe.runner_backend=='target-b12x'
fixed,tree=extract(source)
cases=[]
for dp in [False,True]:
  for explicit in [False,True]:
    for raises in [False,True]:
      draft_backend='draft-flashinfer' if explicit else 'target-b12x'
      flags.moe.runner_backend='target-b12x';events.clear()
      try:
        with fixed(SimpleNamespace(_draft_dp_context_enabled=dp)):
          assert flags.moe.runner_backend==draft_backend
          with fixed(SimpleNamespace(_draft_dp_context_enabled=False)):
            assert flags.moe.runner_backend==draft_backend
          assert flags.moe.runner_backend==draft_backend
          if raises: raise RuntimeError('controlled draft failure')
      except RuntimeError as exc:
        assert raises and str(exc)=='controlled draft failure'
      assert flags.moe.runner_backend=='target-b12x'
      assert events==([('enter_dp','draft-group'),('exit_dp','draft-group')] if dp else [])
      cases.append(dict(dp=dp,explicit=explicit,exception=raises,status='PASS'))

parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
protected=[]
for node in ast.walk(tree):
  if not isinstance(node,ast.Call):continue
  call=ast.unparse(node.func)
  if call not in ['build_draft_tp_worker','self._draft_worker.init_cuda_graphs','self._proposer.propose','self._proposer.run_idle_participation']:continue
  cur=node;covered=False
  while cur in parents:
    cur=parents[cur]
    if isinstance(cur,ast.With) and any(ast.unparse(x.context_expr)=='self._draft_context()' for x in cur.items):covered=True
  assert covered,call
  protected.append(call)
assert len(protected)==4,protected
result=dict(status='PASS',cases=cases,original_mismatch_reproduced=True,protected_calls=protected,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    limit='Source-extracted control-flow checks with metadata fixtures; actual selected model methods and GPU execution require serving validation.')
(Path.cwd()/'context-test-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
