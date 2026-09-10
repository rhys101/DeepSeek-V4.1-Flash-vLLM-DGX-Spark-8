import copy
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cluster', ROOT/'scripts/cluster.py')
cluster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cluster)


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'configs/cluster.example.json').read_text())
        self.config['profile'] = str(ROOT/'configs/profiles/dspark-300k.json')

    def load(self, config):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'config.json'
            p.write_text(json.dumps(config))
            return cluster.load_config(p)

    def test_serving_profile_loads(self):
        self.assertEqual(len(self.load(self.config)['nodes']), 8)

    def test_300k_vision_flag_reaches_node_environment(self):
        self.config['profile'] = str(ROOT/'configs/profiles/dspark-300k.json')
        c = self.load(self.config)
        env = cluster.env_text(c, c['nodes'][0])
        p = subprocess.run(['bash', '-c', env+'printf "%s" "$ENABLE_VISION"'],
                           capture_output=True, text=True, check=True)
        self.assertEqual(p.stdout, '1')

    def test_duplicate_rank_rejected(self):
        self.config['nodes'][1]['rank'] = 0
        with self.assertRaises(ValueError): self.load(self.config)

    def test_duplicate_address_rejected(self):
        self.config['nodes'][1]['fabric_ip'] = self.config['nodes'][0]['fabric_ip']
        with self.assertRaises(ValueError): self.load(self.config)

    def test_shell_injection_in_login_rejected(self):
        self.config['nodes'][1]['ssh_user'] = 'operator; touch /tmp/unsafe'
        with self.assertRaises(ValueError): self.load(self.config)

    def test_model_traversal_rejected(self):
        self.config['model_subpath'] = '../outside'
        with self.assertRaises(ValueError): self.load(self.config)

    def test_bad_image_id_rejected(self):
        self.config['expected_image_id'] = 'latest'
        with self.assertRaises(ValueError): self.load(self.config)

    def test_env_values_are_literal(self):
        self.config['model_store'] = '/srv/models with spaces/$(do-not-execute)'
        c = self.load(self.config)
        env = cluster.env_text(c, c['nodes'][0])
        p = subprocess.run(['bash', '-c', env+'printf "%s" "$MODEL_STORE"'],
                           capture_output=True, text=True, check=True)
        self.assertEqual(p.stdout, self.config['model_store'])

    def test_worker_ssh_is_source_bound(self):
        c = self.load(self.config)
        cmd = cluster.node_command(c, c['nodes'][1], 'true')
        self.assertEqual(cmd[:3], ['ssh', '-b', c['nodes'][0]['fabric_ip']])
        self.assertIn('operator@192.0.2.12', cmd)
        self.assertNotIn('StrictHostKeyChecking=no', cmd)

    def test_300k_allocator_environment_is_preserved(self):
        self.config['profile'] = str(ROOT/'configs/profiles/dspark-300k.json')
        c = self.load(self.config)
        env = cluster.env_text(c, c['nodes'][0])
        p = subprocess.run(['bash', '-c', env+'printf "%s %s %s" "$NCCL_CUMEM_ENABLE" "$NCCL_NVLS_ENABLE" "$PYTORCH_CUDA_ALLOC_CONF"'],
                           capture_output=True, text=True, check=True)
        self.assertEqual(p.stdout, '0 0 expandable_segments:True')

    def test_api_probe_follows_binding(self):
        c = self.load(self.config)
        for host, expected in [('0.0.0.0', 'http://127.0.0.1:8000'),
                               ('192.0.2.11', 'http://192.0.2.11:8000'),
                               ('::', 'http://[::1]:8000'),
                               ('2001:db8::1', 'http://[2001:db8::1]:8000')]:
            c['api_host'] = host
            self.assertEqual(cluster.api_base(c), expected)


if __name__ == '__main__':
    unittest.main()
