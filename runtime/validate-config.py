"""Parse the exact command and validate model configuration without loading weights."""
import json
import sys
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

parser = make_arg_parser(FlexibleArgumentParser())
args = parser.parse_args(sys.argv[1:])
if getattr(args, 'model_tag', None) is not None:
    args.model = args.model_tag
engine = AsyncEngineArgs.from_cli_args(args)
cfg = engine.create_engine_config()
print(json.dumps(dict(
    status='configuration_validated_no_weights_loaded',
    model=cfg.model_config.model,
    architecture=cfg.model_config.hf_config.architectures,
    tensor_parallel=cfg.parallel_config.tensor_parallel_size,
    pipeline_parallel=cfg.parallel_config.pipeline_parallel_size,
    data_parallel=cfg.parallel_config.data_parallel_size,
    max_model_len=cfg.model_config.max_model_len,
    cache_dtype=cfg.cache_config.cache_dtype,
    block_size=cfg.cache_config.block_size,
    max_num_seqs=cfg.scheduler_config.max_num_seqs,
    engram=cfg.engram_config,
    compilation_mode=cfg.compilation_config.mode,
    cudagraph_mode=cfg.compilation_config.cudagraph_mode,
    cudagraph_capture_sizes=cfg.compilation_config.cudagraph_capture_sizes,
    speculative_method=cfg.speculative_config.method if cfg.speculative_config else None,
    speculative_tokens=cfg.num_speculative_tokens,
    draft_architecture=cfg.speculative_config.draft_model_config.architectures if cfg.speculative_config else None,
    draft_tp=cfg.speculative_config.draft_tensor_parallel_size if cfg.speculative_config else None,
), default=str, indent=2))
