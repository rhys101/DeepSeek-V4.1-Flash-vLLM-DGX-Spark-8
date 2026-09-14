"""An explicit 128x4 MXFP8 input contract for the TP8 WO-B projection."""
from typing import NamedTuple
import torch


class SparkMxfp8SwizzledInput(NamedTuple):
    """E4M3 values and UE8M0 group-32 scales in FlashInfer 128x4 order."""
    data: torch.Tensor
    scales: torch.Tensor
