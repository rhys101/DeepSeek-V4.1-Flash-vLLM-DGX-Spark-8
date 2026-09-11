"""Route SGLang's block-fp8-as-MXFP8 dense linears to a FlashInfer backend that
suits GB10 (SM121).

SGLang's --fp8-gemm-backend only knows the cutlass / cute-dsl / trtllm MXFP8
kernels. The CUTLASS SM120 kernel uses a 128x32x128 tile: at M=6 (DSpark verify
at batch 1) it pads M to 128 and streams 32 weight columns per tile, which the
2026-09-10 profile measured at 50-75 GB/s on every projection (52 ms of a 118 ms
decode step, the same as the Triton kernel it replaced). FlashInfer's b12x
warp-level MMA kernel has 16|32 x 64|128 tiles for small M and is what
mm_mxfp8's own 'auto' heuristic prefers on SM120/121; 'cudnn' is the other
SM12x option. DSV41_MXFP8_BACKEND selects it (default b12x; 'cutlass' or ''
restores SGLang's choice). A shape the chosen backend rejects falls back to
cutlass for the rest of the process, logged once.
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)

_REJECTED = set()


def install(module):
    want = os.environ.get('DSV41_MXFP8_BACKEND', 'b12x').strip()
    if want in ('', 'cutlass', 'off', '0'):
        return
    original = module.flashinfer_mxfp8_blockscaled_linear

    def linear(input, weight, weight_scale, input_scale=None, bias=None,
               output_dtype=None, backend='cutlass', pin_tactic=False):
        key = (int(weight.shape[0]), int(weight.shape[1]))
        if backend == 'cutlass' and key not in _REJECTED:
            try:
                return original(input, weight, weight_scale, input_scale, bias,
                                output_dtype, backend=want, pin_tactic=pin_tactic)
            except Exception as exc:  # shape/backend rejection: keep serving
                _REJECTED.add(key)
                logger.warning('DSV41 MXFP8 backend %r rejected N=%s K=%s (%s); '
                               'cutlass for this shape', want, key[0], key[1], exc)
        return original(input, weight, weight_scale, input_scale, bias,
                        output_dtype, backend=backend, pin_tactic=pin_tactic)

    module.flashinfer_mxfp8_blockscaled_linear = linear

    logger.warning('DSV41 MXFP8 dense linears routed to FlashInfer backend %r', want)
