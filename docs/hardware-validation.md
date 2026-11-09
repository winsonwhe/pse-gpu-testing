# Hardware validation required

All automated development verification uses CPU-only fixtures and command plans.
PDFs named in config/comments were not available. The following checks need real
captures and execution on each supported SKU/network mode before production use.

| Check | Verify on hardware |
|---|---|
| `network_mode` | Actual OCI IMDS schema/types, absent/1 vs 4 planes, access from every target, mixed-mode rejection and non-OCI fallback. |
| `nvidia_smi` | GPU count/model/memory, SKU-specific driver minima, ECC reporting, partial A10 policy and query availability. Confirm configured B300 shape vs GB300 distinctions; CUDA toolkit minimum is not currently enforced. |
| `gpu_topology` | Actual topo matrix header and NIC enumeration, virtual-function vs physical-NIC counts, missing devices and driver-dependent formats. Presence does not prove bandwidth/affinity correctness. |
| `fabricmanager` | Whether the deployment uses this service, its matching driver version, permissions and failure behavior on NVSwitch hosts. |
| `numa_topology` | Grace CPU/memory node layout, actual counts and zero-MB nodes; x86 alternatives. |
| `rdma_iface` / `rdma_link` | Single-plane device counts/names/netdev mapping on every shape; multiplanar four VF rails, VRF presence, iproute2 master JSON formats and correct VF-to-VRF binding. Ignore unused DOWN entries. |
| `ssh_connectivity` / executor | Controller and launcher-to-peer keys, trusted host keys, timeouts, cancellation/disconnection cleanup and remote command permissions. SSH setup is not full mesh provisioning. |
| `gpu_burn` | CUDA 10.3 compilation and runtime PTX, numerical error handling, one-hour stability, throughput balance and baseline tolerance, memory selection, clocks/cooling/power, telemetry sampling. Validate optional fryer CLI/NVML paths; fryer cannot certify numerical correctness. Absolute thermal/power limits are not enforced. |
| `nvbandwidth` | Exact installed options/test names, native SUM output and units, build flags, P2P/NVLink availability, SKU floors. Multi-node mode additionally needs one GMF, shared paths, IMEX, MPI binding and all selected GPU ranks. |
| `babelstream` | CUDA architecture/build options, device selection, each native output/units, all five bandwidth floors and GPU-to-GPU spread under sustained load. |
| `dcgmi` | Installed DCGM JSON schema, selected r1/r2/r3/r4 support, runtime/permissions, entity-level warnings/errors and idle-GPU requirements. Synthetic level fixtures do not prove hardware test coverage. |
| `rdma_perf` / `rdma_bw` / `rdma_lat` | Real perftest reports and average columns, server readiness/cleanup/control ports, matched rail ordering, MTU/address/GID configuration and routability. Verify supplied multiplanar rail0/1 -> NUMA0, rail2/3 -> NUMA1 against the missing PDF; verify VRF/IPv6/GID3 commands. Calibrate ~459 Gb/s / 3.8–4.0 us examples and wide tolerances. Single-plane family/GID/tuning and thresholds remain deployment-specific. |
| `placement` | Actual GMF/local-block relationships, expected rack size (up to 18), selection within one GMF, preference across racks and unavailable/mismatched metadata. |
| `imex` | Membership names/IPs, per-GMF configuration on all hosts, service state and actual transport health. No cross-rack IMEX domain; state/membership alone do not prove transport. |
| `nccl_allreduce` / `nccl_alltoall` | nccl-tests options/output, all ranks and 16 GiB completion, both placements, correctness counters, spcx/UCX/HPCX availability, interface/GID/TC/env values, <=32 vs >32 tuning, MNNVL/NVLS and per-GMF IMEX. Calibrate each SKU/mode/scale; PDF scale tables are deliberately empty. Verify mpirun/Open MPI or HPC-X launching, host slots and exported environment on the actual hosts. |
| `baseline` / reports / profiles | Obtain operator-confirmed healthy runs at each profile/scale; calibrate tolerances and test repeatability under comparable clocks/load. Validate observed regressions against genuine degradations. Check profile wall time with prebuilt tools; real builds/diagnostics can exceed target budgets. Mock and real snapshots are incompatible. |

Official references:

- [OCI GB300 guidance](https://github.com/oracle-quickstart/oci-gpu-quickstarts/blob/main/nvidia/GB300/README-GB300.md)
- [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)
- [DCGM diagnostics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html)
- [NVIDIA IMEX guide](https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/overview.html)
