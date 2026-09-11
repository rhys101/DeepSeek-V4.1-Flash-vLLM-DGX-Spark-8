"""Run EP4 native-head and verify/index numerical tests on an idle rank-zero GPU."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import cluster

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',default=str(cluster.ROOT/'configs/cluster.local.json'))
args=p.parse_args();c=cluster.load(args.config)
cluster.require_head(c);cluster.expected(c)
cluster.image_check(c,c['workers'][0]);cluster.mapped(c,lambda n:cluster.idle(c,n))
out=Path(c['run_dir'])/'state/kernel-check';out.mkdir(exist_ok=False)
tests=[
    '/validation/upstream-tests/registered/kernels/ops/attention/dsv4/test_c2_verify.py',
    '/validation/upstream-tests/registered/kernels/test_dsv4_indexer_postprocess.py',
    '/validation/upstream-tests/registered/attention/unittests/dsv4/test_dsv41_fused_compress.py',
]
for name,argv in [('native',['-u','/validation/check-native-heads.py']),
                  ('verify',['-m','pytest','-q','-ra','--tb=short',*tests,'--junitxml=/state/kernel-check/junit.xml'])]:
    cmd=cluster.command(c,0,'dense');cmd[-1:]=argv
    (out/(name+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
    try:
        with (out/(name+'.log')).open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
    except BaseException:
        subprocess.run(['docker','stop','-t','10',cmd[cmd.index('--name')+1]],check=False,timeout=30)
        raise
    print(name+' kernel checks passed',flush=True)
print('EP4_KERNEL_CHECKS_PASS')
