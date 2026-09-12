"""Pure configuration rendering; safe to inspect without Torch or a GPU."""
import ipaddress
import json
from pathlib import Path
import re

def load(path):
    c = json.loads(Path(path).read_text())
    nodes = sorted(c['workers'], key=lambda n: n['rank'])
    if len(nodes) != 8 or [n['rank'] for n in nodes] != list(range(8)):
        raise ValueError('Exactly eight ranks 0..7 are required')
    ips = [str(ipaddress.IPv4Address(n['ip'])) for n in nodes]
    if len(set(ips)) != 8 or ips[0] != c['head_ip']:
        raise ValueError('Unique node IPs and rank-0 head_ip are required')
    for n in nodes:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+', n['host']):
            raise ValueError('Invalid SSH destination')
    for key in ('model_store', 'run_dir'):
        p = Path(c[key])
        if not p.is_absolute() or '..' in p.parts or any(x in str(p) for x in '\n\r\0'):
            raise ValueError('Use absolute model and deployment paths without traversal')
    sub = Path(c['model_subpath'])
    if sub.is_absolute() or '..' in sub.parts or not sub.parts:
        raise ValueError('Invalid model subdirectory')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', c['fabric_interface']): raise ValueError('Invalid fabric interface')
    if not re.fullmatch(r'[=A-Za-z0-9_:,.-]+', c['nccl_hcas']): raise ValueError('Invalid HCA selection')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/:@-]*', c['image']): raise ValueError('Invalid image reference')
    ipaddress.ip_address(c['api_host'])
    for key in ('dist_port', 'api_port', 'nccl_test_port'):
        if type(c[key]) is not int or not 1024 <= c[key] <= 65535: raise ValueError('Invalid port')
    if c['context_length'] != 1000000 or c['max_running_requests'] != 128 or c['draft_length'] != 5:
        raise ValueError('This comparison profile is fixed at 1M context, 128 request slots and Mia’s DSpark draft length 5')
    if c['chunked_prefill_size'] not in (1024, 2048, 4096, 8192, 16384): raise ValueError('Unsupported prefill chunk')
    if not .65 <= c['mem_fraction_static'] <= .80: raise ValueError('Keep memory fraction at or below 0.80 for initial validation')
    if c['max_total_tokens'] != 8000000: raise ValueError('This capacity test requires an 8M logical-token pool')
    if c['minimum_available_gib'] != 2: raise ValueError('This capacity test requires the user-authorized 2 GiB OS reserve')
    if c.get('min_free_slots_delay',1) != 1: raise ValueError('Only disabling refill waiting is supported by this profile')
    if c.get('ep_size',8) not in (1,2,4,8): raise ValueError('EP must divide TP8')
    if c.get('nccl_channels',8) not in (1,2,4,8,16): raise ValueError('Unexpected NCCL channel count')
    if c.get('nccl_algo','') not in ('','Ring','Tree'): raise ValueError('Unexpected NCCL algorithm')
    if c['max_prefill_tokens'] != c['chunked_prefill_size']: raise ValueError('Keep the two prefill budgets aligned')
    if c.get('cuda_graph_batch_sizes') != [1,2,3,4,5,6,7,8,12,16,24,32,48,64,96,128]: raise ValueError('Unexpected capture sizes')
    c['workers'] = nodes
    return c

def engine_args(c, rank):
    if not 0 <= rank < 8: raise ValueError('Invalid rank')
    return (['--min-free-slots-delay', str(c['min_free_slots_delay'])] if 'min_free_slots_delay' in c else []) + [
        '--model-path', '/models/' + c['model_subpath'],
        '--served-model-name', 'deepseek-v41-flash',
        '--load-format', 'safetensors', '--dtype', 'bfloat16', '--trust-remote-code',
        '--tp-size', '8', '--ep-size', str(c.get('ep_size',8)), '--nnodes', '8', '--node-rank', str(rank),
        '--dist-init-addr', f"{c['head_ip']}:{c['dist_port']}",
        '--attention-backend', 'dsv4', '--moe-runner-backend', 'flashinfer_mxfp4',
        '--speculative-moe-runner-backend', 'flashinfer_mxfp4',
        '--fp8-gemm-backend', 'flashinfer_cutlass', '--kv-cache-dtype', 'auto',
        '--context-length', str(c['context_length']),
        '--max-running-requests', str(c['max_running_requests']),
        '--chunked-prefill-size', str(c['chunked_prefill_size']),
        '--max-prefill-tokens', str(c['max_prefill_tokens']),
        '--max-total-tokens', str(c['max_total_tokens']),
        '--mem-fraction-static', str(c['mem_fraction_static']),
        '--cuda-graph-max-bs-decode', '128', '--cuda-graph-bs-decode', *map(str, c['cuda_graph_batch_sizes']),
        '--enable-decoder-swa-bounded-replay', '--disable-custom-all-reduce',
        '--speculative-algorithm', 'DSPARK', '--speculative-dspark-block-size', str(c['draft_length']),
        '--speculative-accept-threshold-single', '1.0', '--speculative-accept-threshold-acc', '1.0',
        '--random-seed', '0', '--enable-multimodal',
        '--limit-mm-data-per-request', '{"image":4,"video":0,"audio":0}',
        '--default-chat-template-kwargs', '{"thinking":false}',
        '--tool-call-parser', 'deepseekv41', '--reasoning-parser', 'deepseek-v41',
        '--enable-cache-report', '--watchdog-timeout', '1800',
        '--host', c['api_host'], '--port', str(c['api_port']),
    ]

def environment(c, rank):
    env = {
        'SGLANG8_RESIDENT_PROFILE': '1',
        'PYTHONPATH': '/opt/sglang8/adapter:/opt/sglang8/runtime:/opt/sglang8/b12x',
        'SGLANG8_ROCE_ALLREDUCE': '1',
        'B12X_ROCE_CACHE_DIR': '/cache/roce-sg17',
        'B12X_COMPILE_CACHE_DIR': '/cache/b12x-sg17',
        'CUTE_DSL_CACHE_DIR': '/cache/cute-sg17',
        'SGLANG8_RETRY_FREE_ADMISSION': '1' if c.get('retry_free_admission',False) else '0',
        'SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE': '0',
        'SGLANG_ENABLE_DSV41_ENGRAM_KV_PREFETCH': '0',
        'DSV41_TP_PAD': '0', 'DSV41_MXFP8_BACKEND': 'b12x',
        'DSV41_PREFILL_EMPTY_CACHE_TOKENS': '8192',
        'SGLANG_FLASHINFER_MOE_FUSED_FINALIZE': '0',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:False',
        'SGLANG_RAGGED_VERIFY_MODE': 'static',
        'NCCL_SOCKET_IFNAME': '=' + c['fabric_interface'],
        'GLOO_SOCKET_IFNAME': c['fabric_interface'], 'NCCL_IB_HCA': c['nccl_hcas'],
        'NCCL_IB_DISABLE': '0', 'NCCL_IB_ADDR_FAMILY': 'AF_INET', 'NCCL_NET': 'IB',
        'NCCL_CUMEM_ENABLE': '0', 'NCCL_NVLS_ENABLE': '0',
        'NCCL_BUFFSIZE': '1048576', 'NCCL_LL128_BUFFSIZE': '262144',
        'NCCL_PROTO': '^LL128', 'NCCL_MAX_NCHANNELS': str(c.get('nccl_channels',8)),
        'NCCL_DEBUG': 'INFO', 'NCCL_DEBUG_SUBSYS': 'INIT,NET',
        'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HOME': '/cache/huggingface', 'XDG_CACHE_HOME': '/cache',
        'TRITON_CACHE_DIR': '/cache/triton', 'FLASHINFER_CUDA_ARCH_LIST': '12.1a',
        'TORCH_CUDA_ARCH_LIST': '12.1a', 'MAX_JOBS': '2',
    }

    if c.get('nccl_algo'): env['NCCL_ALGO']=c['nccl_algo']
    return env
