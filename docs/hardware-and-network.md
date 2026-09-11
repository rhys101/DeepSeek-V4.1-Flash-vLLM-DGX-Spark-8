# Hardware and network observations

The recorded deployment uses eight GB10 nodes and two RoCE interfaces per node. NCCL selected its IB/RoCE payload transport. Bootstrap sockets alone do not establish which transport carries tensor data; inspect the `NET/IB` and selected-network messages.

MTU was 9000 on all 16 fabric interfaces. A non-fragmenting 8972-byte IPv4 ICMP payload succeeded from rank 0 to every peer over each fabric. The deployment scripts bind SSH and rsync to the selected rank-0 fabric address and check its route before copying or launching.

Interface names and RDMA device names are machine-specific. `NCCL_IB_HCA` takes RDMA HCA names; it is not the Ethernet interface name. Set both explicitly in `cluster.local.json`. The supplied names are examples from the tested hardware.

GPUDirect RDMA was reported disabled by NCCL. The available evidence establishes RoCE transport and correct collectives, not GPUDirect operation. The setup does not change host drivers, MTU, clock policy, swap, or kernel settings automatically.

During the base-stack benchmark on 10 September 2026, twenty samples per GPU showed SM clocks of 2,171–2,190 MHz, P0 and no active reported throttle flags. These observations do not prove the absence of performance-state changes between samples. GPU utilization can include time waiting in collectives; GPU power is not total node power.

The model-unloaded GPU-state probe completed separately on all eight nodes without a large collapse. It does not establish the resident model’s state throughout serving. [Raw clock and probe evidence](../results/base-stack-validation/).
