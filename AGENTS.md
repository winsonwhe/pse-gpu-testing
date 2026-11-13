# gputest: GPU Validation Toolkit

## Goal
A small, practical automation tool to validate NVIDIA B300 / GB300 hosts and other GPU shapes (Grace ARM, aarch64; GB300 has multi-host NVLink72).
The stress-test idea comes from https://github.com/wilicc/gpu-burn, extended to cover: preflight checks, GPU burn, NVLink / memory bandwidth, DCGM diagnostics, RDMA, NCCL.

References:
- https://github.com/oracle-quickstart/oci-gpu-quickstarts/blob/main/nvidia/GB300/README-GB300.md
- ./docs/*

## Language
All project content is in English: code, comments, CLI help, log messages, reports, README, docs, commit messages.

## Design Rules (keep it simple)
- Python 3.10+, standard library + PyYAML only.
- Flat layout, few files. Do NOT add packages, base-class hierarchies, or plugin frameworks.
- A test is a plain function registered with a decorator, e.g. `@check("nvidia_smi", category="preflight")`.
  Adding a test = add one function + (optionally) one block in `config.yaml`.
- Every external command goes through one `run()` helper: timeout, dry-run, raw output saved to disk, mock mode (reads from `fixtures/`).
- Parsers are pure functions (text in, dict out) and have unit tests built from real sample output.
- Thresholds, version requirements, tool paths and NCCL env vars live in `config.yaml`, never hardcoded.
- Result status: PASS / WARN / FAIL / SKIP. Always keep the raw output.
- Must be testable on a machine with no GPU (pytest + fixtures).

## Target Layout
```
gputest/
  cli.py          # argparse CLI: list / run / report
  core.py         # Result, @check registry, run() helper, node executor (local / ssh), report writer
  checks.py       # all test functions (preflight, burn, bandwidth, dcgm, rdma, nccl)
  parsers.py      # pure parsing functions
config.yaml       # profiles (quick/standard/full), sku settings, network modes, thresholds, NCCL env
fixtures/         # sample command outputs for mock mode and unit tests
tests/            # pytest
docs/             # reference PDF
README.md
```
Split a file only when it grows past roughly 600 lines.

## Network Mode: Single-Plane vs Multiplanar (important)
The OCI shape name `BM.GPU.GB300.4` is the same for both variants. Never infer the network type from the shape name.

- Detect it from IMDS: `curl -H "Authorization: Bearer Oracle" -L http://169.254.169.254/opc/v2/host`, read `rdmaFabricData.planes`.
  - `planes == 4` -> multiplanar
  - field missing or `planes == 1` -> single-plane (confirm exact IMDS values against the OCI README)
  - IMDS unreachable (non-OCI) -> use `network.mode` from config
- `network.mode: auto | single_plane | multiplanar` in `config.yaml` overrides detection.
- All network-dependent checks (rdma_iface, rdma_perf, nccl env) branch on the detected mode. Keep mode-specific expectations in `config.yaml`, not in code.

Multiplanar facts (from the OCI manual):
- Expect `rdma_vf_rail0..3` (customer-facing VFs, must be ACTIVE + LINK_UP), `vrf_r0..3` (each VF enslaved to its VRF),
  16 per-plane netdevs `rdma_p<P>_rail<R>` (P, R in 0..3), `rdma0..3`, `br-rail0..3`.
- Many `rdma link` entries showing DOWN / DISABLED are normal; ignore them when active links exist.
- RDMA perf runs through VRF + IPv6: `ip vrf exec vrf_rX numactl -N <node> ib_write_bw ... --ipv6 --ipv6-addr -x 3 -d rdma_vf_railX`
  (rail0/1 -> NUMA 0, rail2/3 -> NUMA 1; verify against the PDF before implementing).
- NCCL: `NCCL_IB_HCA=rdma_vf_rail0,rdma_vf_rail1,rdma_vf_rail2,rdma_vf_rail3`.

Single-plane facts:
- The PDF does NOT describe single-plane interface names, VRFs, or addressing. Do not invent them.
- Approach: discover active RDMA devices from `rdma link` (state ACTIVE, physical_state LINK_UP), expect one per GPU (4 per host, configurable),
  map each to its netdev, and let `config.yaml` optionally pin device names and IP family.
- Read the OCI README linked above for the real single-plane details, and mark anything unverified with `TODO(verify)`.
- Single-plane RDMA perf and NCCL use the discovered devices; do not assume VRF or IPv6 unless detected.

## Hardware / Environment Facts (OCI manual)
- Per host: 2 x Grace (72 cores each, 144 total), 960 GB memory, 4 x GB300/B300 GPUs, 4 x ConnectX-8 800 Gb/s RDMA NICs (one per GPU), 2 x ConnectX-7 200 Gb/s frontend.
- NUMA: `numactl --hardware` shows ~34 nodes; only nodes 0 and 1 have CPU + memory. Ignore the 0 MB nodes.
- Recommended versions: DOCA OFED 3.1.0+, NVIDIA driver 580.x+, CUDA 13+, NCCL 2.28.7+, HPCX 2.24.1+.
- Useful IMDS fields: `rdmaTopologyData.customerGpuMemoryFabric` (GMF / rack / NVLink domain), `customerLocalBlock` (Deployment Group), `customerHostId`.
- IMEX is per GMF/rack (max 18 hosts), GB300 only. B300 single host: SKIP.
- Multi-host placement: single-rack jobs stay inside one GMF; cross-rack jobs prefer GMFs sharing one `customerLocalBlock`.

## Working Rules
- If a parameter is uncertain (`sm_100`, nvbandwidth options, dcgmi output format, single-plane details), check official docs, cite the source in a comment, or leave `TODO(verify)`. Do not guess.
- Commands with `sudo`, SSH, or multi-host effects must support `--dry-run` that prints the commands first.
- After each task: run `pytest` and `ruff`, and list what could not be verified without real GPU hardware.
- Prefer deleting code over adding abstraction. Ask before large restructuring.
