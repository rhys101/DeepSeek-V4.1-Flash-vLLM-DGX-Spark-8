"""Mia-derived kernel hooks, with native resident Engram and no TP3 padding."""
import importlib.abc
import importlib.machinery
import os
import sys

TARGETS = {
    'sglang.srt.layers.engram',
    'sglang.srt.managers.scheduler',
    'sglang.srt.layers.quantization.fp8_utils',
    'sglang.srt.model_executor.model_runner',
    'sglang.srt.layers.attention.dsv4.metadata',
}

class Loader(importlib.abc.Loader):
    def __init__(self, original): self.original = original
    def create_module(self, spec): return self.original.create_module(spec)
    def exec_module(self, module):
        self.original.exec_module(module)
        if module.__name__.endswith('.scheduler'):
            from admission import install
            install(module)
        elif module.__name__.endswith('.engram'):
            from resident_engram import install
            install(module)
        elif module.__name__.endswith('.fp8_utils'):
            from mxfp8_b12x import install
            install(module)
        elif module.__name__.endswith('.model_runner'):
            from prefill_empty_cache import install
            install(module)
        else:
            cls = module.PagedIndexerMetadata
            original = cls.__post_init__
            def post_init(self):
                if bool(getattr(module, '_IS_SM120', False)) and self.compress_ratio in (1, 2):
                    self.force_deep_gemm_metadata = True
                original(self)
            cls.__post_init__ = post_init

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS: return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None: spec.loader = Loader(spec.loader)
        return spec

if os.environ.get('SGLANG8_RESIDENT_PROFILE') == '1':
    sys.meta_path.insert(0, Finder())
