The original files are project-provided samples. New format fixtures are offline
examples, not measurements taken on a B300/GB300 host:

- `gpu_burn_progress.txt`: progress-vector layout from
  https://github.com/wilicc/gpu-burn/blob/master/gpu_burn-drv.cpp .
- `gpu_fryer.txt`: the real example row from
  https://github.com/huggingface/gpu-fryer#usage, repeated for four GPU IDs.
- `babelstream_native.txt`: existing project sample values converted to the native
  MB/s table printed by https://github.com/UoB-HPC/BabelStream/blob/main/src/main.cpp .
- `nvbandwidth_gb300.txt`: illustrative passing aggregate SUM for mock mode,
  not a GB300 measurement or a calibrated performance target.
- `nvbandwidth_native.txt`: named SUM syntax from
  https://github.com/NVIDIA/nvbandwidth#usage, with illustrative numeric totals.
- `dcgmi_r1.txt`, `dcgmi_r3.txt`, `dcgmi_r4.txt`: minimal illustrative
  passing software results for profile mock runs, using the same NVIDIA schema.
- `dcgmi_native.json`: representative outcomes in the NVIDIA schema at
  https://github.com/NVIDIA/DCGM/blob/master/testing/python3/dcgm_diag_schema.json .
- `gpu_samples.txt`: illustrative nvidia-smi CSV samples for peak aggregation.

Replace illustrative examples with raw hardware captures when available. These
fixtures test parsing and decision logic, not hardware performance or compatibility.

Multi-host fixtures:

- `nccl_allreduce.txt` and `nccl_alltoall.txt` are verbatim result-table excerpts
  from the OCI GB300 README (retrieved 2026-11-09):
  https://github.com/oracle-quickstart/oci-gpu-quickstarts/blob/main/nvidia/GB300/README-GB300.md
  The all-reduce table is the two-host example; the all-to-all table is the
  single-host example. They validate parsing and are not scale reference baselines.
- `ib_write_bw.txt` and `ib_write_lat.txt` are synthetic perftest-format samples
  around the user-supplied multiplanar PDF example values. The PDF is unavailable
  in this checkout; TODO(verify): replace these with raw hardware captures.
  Column layout follows https://github.com/linux-rdma/perftest.
- `rdma_link_multiplanar.txt`, `rdma_address.json`, `rdma_listen.txt`,
  `imds_host.json`, `imex_nodes.txt`, and `imex_service.txt` are synthetic discovery
  and service-state data. `n1` and `n2` are mock hosts in one GMF.
- `rdma_link_single_plane.txt` reuses `rdma_link_a10.txt`; no new single-plane
  device naming claims are made.

Project completion fixtures: `build.txt`, `fabricmanager_active.txt`,
`vrf_links.json` and `gpu_topology_<sku>.txt` are synthetic success plans, not
hardware captures. The topology samples contain only the matrix header needed
for presence parsing; actual hardware matrices still need verification.
`numactl_gb300.txt` and `numactl_b300.txt` reuse the two-CPU-node A10 sample to
exercise filtering and do not assert Grace memory/core counts.
