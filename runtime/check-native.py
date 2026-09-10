"""Run selected unchanged upstream tests for the rebuilt DeepSeek extension."""
import json
import runpy
import sys
import torch

# This mount contains only upstream tests, so installed vLLM remains selected.
sys.path.insert(0,'/opt/spark8-tests')
tests=runpy.run_path('/opt/spark8-tests/tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py')
assert tests['_op_available'](), 'The rebuilt fused DeepSeek kernel is absent'
torch.set_num_threads(2)
for n in (4,48,2048):
    tests['test_q_path_without_qnorm_matches_rope_only_reference'](n)
    print(json.dumps(dict(status='PASS',case='upstream_rope_without_qnorm',tokens=n)),flush=True)
    tests['test_combined_q_and_kv'](n,8,8,64)
    print(json.dumps(dict(status='PASS',case='upstream_fused_q_and_kv',tokens=n,heads=8,page=64)),flush=True)
