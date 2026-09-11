"""CPU checks for immutable overlay installation and the public EP4 launch."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import cluster
spec = importlib.util.spec_from_file_location('apply_overlays', ROOT / 'scripts/apply-overlays.py')
overlays = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlays)

class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/cluster.example.json').read_text())

    def load(self, changes=None):
        c = copy.deepcopy(self.config)
        c.update(changes or {})
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'profile.json'
            p.write_text(json.dumps(c))
            return cluster.load(p)

    def test_all_ranks_render_validated_engine_configuration(self):
        c = self.load()
        for rank in range(8):
            args = cluster.engine_args(c, rank)
            for key, expected in {'--tp-size':'8','--ep-size':'4','--nnodes':'8','--node-rank':str(rank),
                                  '--min-free-slots-delay':'1','--speculative-dspark-block-size':'5',
                                  '--moe-runner-backend':'flashinfer_mxfp4','--context-length':'300000',
                                  '--max-running-requests':'8','--max-total-tokens':'3200000'}.items():
                self.assertEqual(args[args.index(key)+1], expected)
            env = cluster.environment(c, rank)
            self.assertEqual(env['SGLANG8_RETRY_FREE_ADMISSION'], '0')
            self.assertEqual(env['SGLANG_FLASHINFER_MOE_FUSED_FINALIZE'], '0')
            self.assertEqual(env['SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE'], '0')
            cmd = cluster.command(c, rank)
            self.assertIn('type=bind,src=/srv/models,dst=/models,readonly', cmd)
            self.assertEqual(cmd[cmd.index('--name')+1], f'sglang8-ep4-r{rank}')

    def test_unvalidated_profile_changes_are_rejected(self):
        for change in [dict(ep_size=8), dict(draft_length=3), dict(min_free_slots_delay=2),
                       dict(retry_free_admission=True), dict(minimum_available_gib=12),
                       dict(context_length=310000), dict(mem_fraction_static=0.9),
                       dict(mem_fraction_static=0.75), dict(max_total_tokens=2400000),
                       dict(chunked_prefill_size=4096,max_prefill_tokens=4096), dict(nccl_channels=4),
                       dict(run_dir='/'), dict(model_store='/srv/models,bad')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(change)

    def test_no_placeholder_image_can_be_launched(self):
        with self.assertRaises(RuntimeError):
            cluster.expected(self.load())

    def test_base_and_overlay_integrity_checks_precede_all_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'patches'; target=root/'original'
            (source/'source').mkdir(parents=True);target.mkdir()
            manifest={}
            for name in ('a.py','b.py'):
                (target/name).write_bytes(b'original')
                (source/'source'/name).write_bytes(b'replacement')
                manifest[name]=dict(before_sha256=hashlib.sha256(b'original').hexdigest(),after_sha256=hashlib.sha256(b'replacement').hexdigest())
            (source/'manifest.json').write_text(json.dumps(manifest))
            (target/'b.py').write_bytes(b'unexpected base')
            with self.assertRaisesRegex(ValueError,'Base source differs'):
                overlays.apply(source,target)
            self.assertEqual((target/'a.py').read_bytes(),b'original')
            (target/'b.py').write_bytes(b'original')
            (source/'source/b.py').write_bytes(b'unexpected replacement')
            with self.assertRaisesRegex(ValueError,'Overlay source differs'):
                overlays.apply(source,target)
            self.assertEqual((target/'a.py').read_bytes(),b'original')
            (source/'source/b.py').write_bytes(b'replacement')
            self.assertEqual(overlays.apply(source,target),2)
            self.assertEqual((target/'a.py').read_bytes(),b'replacement')

    def test_final_source_and_runtime_manifest(self):
        manifest=json.loads((ROOT/'patches/manifest.json').read_text())
        self.assertEqual(len(manifest),8)
        for name,entry in manifest.items():
            self.assertEqual(hashlib.sha256((ROOT/'patches/source'/name).read_bytes()).hexdigest(),entry['after_sha256'])
        lock=json.loads((ROOT/'versions.lock.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT/'runtime/configuration.py').read_bytes()).hexdigest(),lock['runtime_configuration_sha256'])
        for name,expected in lock['adapter_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/'adapter'/name).read_bytes()).hexdigest(),expected)

    def test_layout_validator_matches_actual_all_rank_receipts(self):
        receipts=json.loads((ROOT/'results/ep4/cluster/selected-moe-layout.json').read_text())
        self.assertEqual(len(receipts),8)
        for row in receipts:
            cluster.validate_layout_record(row['layout'],row['rank'])
        changed=copy.deepcopy(receipts[0]['layout'])
        name=next(iter(changed['draft']))
        changed['draft'][name]['method']='Mxfp4FlashinferTrtllmMoEMethod'
        with self.assertRaisesRegex(RuntimeError,'Unexpected expert layout'):
            cluster.validate_layout_record(changed,0)
        with self.assertRaisesRegex(RuntimeError,'Unexpected expert layout'):
            cluster.validate_layout_record(receipts[0]['layout'],1)

    def test_real_source_draft_context_restores_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            result=subprocess.run([sys.executable,str(ROOT/'validation/check-context.py')],cwd=temp,capture_output=True,text=True,check=True)
            record=json.loads((Path(temp)/'context-test-result.json').read_text())
            self.assertEqual(record['status'],'PASS')
            self.assertEqual(len(record['cases']),8)

if __name__=='__main__':unittest.main()
