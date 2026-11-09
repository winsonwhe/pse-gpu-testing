"""All check functions. Import side-effect: registers checks in core._REGISTRY."""

from __future__ import annotations

import socket

from gputest import parsers
from gputest.core import Result, RunContext, check

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

@check("nvidia_smi", category="preflight")
def nvidia_smi(ctx: RunContext) -> Result:
    """Verify GPU count, model, driver, memory size, ECC state."""
    sku = ctx.sku_config()

    # Choose fixture by SKU so mock mode works for both GB300 and A10
    fixture = f"nvidia_smi_{ctx.sku}.txt" if ctx.sku != "gb300" else "nvidia_smi.txt"

    rr = ctx.run(
        ["nvidia-smi",
         "--query-gpu=index,name,driver_version,memory.total,ecc.mode.current",
         "--format=csv,noheader,nounits"],
        mock_fixture=fixture,
    )

    if rr.returncode != 0 and not rr.skipped:
        return Result(
            name="nvidia_smi",
            status="FAIL",
            message=f"nvidia-smi exited {rr.returncode}: {rr.stderr.strip()[:200]}",
            _raw=rr.stdout + rr.stderr,
        )

    parsed = parsers.parse_nvidia_smi_csv(rr.stdout)

    if not parsed:
        if rr.skipped:
            return Result(name="nvidia_smi", status="SKIP", message="dry-run")
        return Result(name="nvidia_smi", status="FAIL",
                      message="no GPU rows parsed from nvidia-smi output",
                      _raw=rr.stdout)

    issues = []
    metrics: dict = {}

    # GPU count
    expected_count = sku.get("gpu_count")
    gpu_count_min = sku.get("gpu_count_min", expected_count)
    actual_count = len(parsed)
    metrics["gpu_count"] = actual_count
    if expected_count and actual_count != expected_count:
        # For A10 a partial host (1-3 GPUs) is acceptable if within [gpu_count_min, gpu_count]
        if gpu_count_min is not None and actual_count < gpu_count_min:
            issues.append(
                f"expected {gpu_count_min}-{expected_count} GPUs, found {actual_count}"
            )
        elif gpu_count_min == expected_count:
            issues.append(f"expected {expected_count} GPUs, found {actual_count}")

    # Model pattern
    model_pattern = sku.get("gpu_model_pattern", "")
    models = [g["name"] for g in parsed]
    metrics["gpu_models"] = models
    if model_pattern:
        bad_models = [m for m in models if model_pattern.upper() not in m.upper()]
        if bad_models:
            issues.append(f"unexpected GPU model(s): {bad_models}")

    # Driver version
    drivers = list({g["driver_version"] for g in parsed})
    metrics["driver_versions"] = drivers
    driver_min = sku.get("driver_min", "0")
    for drv in drivers:
        if _version_lt(drv, driver_min):
            issues.append(f"driver {drv} < minimum {driver_min}")

    # Compute capability (informational)
    cap_fixture = f"nvidia_smi_compute_cap_{ctx.sku}.txt"
    rr2 = ctx.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        mock_fixture=cap_fixture,
    )
    if rr2.returncode == 0 and rr2.stdout.strip():
        caps = [ln.strip() for ln in rr2.stdout.splitlines() if ln.strip()]
        metrics["compute_caps"] = caps

    # Memory size
    mem_expected = sku.get("mem_mib_per_gpu")
    if mem_expected:
        for g in parsed:
            mem = g.get("memory_total_mib")
            if mem is not None and abs(mem - mem_expected) > 512:
                issues.append(
                    f"GPU {g['index']}: memory {mem} MiB, expected ~{mem_expected} MiB"
                )
        metrics["mem_mib_per_gpu"] = mem_expected

    # ECC
    ecc_states = list({g.get("ecc_mode", "N/A") for g in parsed})
    metrics["ecc_modes"] = ecc_states

    if issues:
        return Result(
            name="nvidia_smi",
            status="FAIL",
            message="; ".join(issues),
            metrics=metrics,
            _raw=rr.stdout,
        )

    return Result(
        name="nvidia_smi",
        status="PASS",
        message=f"{actual_count}x {model_pattern}, driver {drivers[0]}, ecc={ecc_states[0]}",
        metrics=metrics,
        _raw=rr.stdout,
    )


@check("ssh_connectivity", category="preflight", requires_multi_host=True)
def ssh_connectivity(ctx: RunContext) -> Result:
    """
    Verify passwordless SSH from this host to every node in --nodes, and back.

    Checks required for NCCL / MPI multi-host runs:
      1. This host → each remote node: `ssh node hostname` succeeds without a password prompt.
      2. Each remote node → this host: `ssh node ssh <local_hostname> hostname` round-trips.
      3. known_hosts is pre-populated for all nodes (BatchMode catches missing host keys).

    Reports per-node PASS/FAIL so the user knows exactly which link is broken.
    Uses `ssh -o BatchMode=yes -o ConnectTimeout=10` so a missing key or
    password prompt fails immediately rather than hanging.
    """
    local_hostname = socket.gethostname()
    failed: list[str] = []
    details: list[str] = []

    for node in ctx.nodes:
        # Forward: this host → node
        rr = ctx.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             "-o", "StrictHostKeyChecking=no", node, "hostname"],
            mock_fixture="ssh_hostname.txt",
        )
        if rr.skipped:
            return Result(name="ssh_connectivity", status="SKIP", message="dry-run")
        if rr.returncode != 0:
            failed.append(node)
            details.append(f"→{node}: FAIL ({rr.stderr.strip()[:120]})")
            continue
        details.append(f"→{node}: PASS ({rr.stdout.strip()})")

        # Reverse: node → this host (needed for MPI reverse connections)
        rr2 = ctx.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             "-o", "StrictHostKeyChecking=no", node,
             "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             "-o", "StrictHostKeyChecking=no", local_hostname, "hostname"],
            mock_fixture="ssh_hostname.txt",
        )
        if rr2.returncode != 0:
            failed.append(f"{node}→{local_hostname}")
            details.append(
                f"{node}→{local_hostname}: FAIL ({rr2.stderr.strip()[:120]})"
            )
        else:
            details.append(f"{node}→{local_hostname}: PASS")

    msg = "; ".join(details)
    if failed:
        return Result(
            name="ssh_connectivity",
            status="FAIL",
            message=f"SSH broken: {', '.join(failed)} | {msg}",
            metrics={"failed_links": failed},
        )
    return Result(
        name="ssh_connectivity",
        status="PASS",
        message=msg,
        metrics={"nodes_checked": len(ctx.nodes)},
    )


@check("fabricmanager", category="preflight", skus=["gb300", "b300"])
def fabricmanager(ctx: RunContext) -> Result:
    """Check nvidia-fabricmanager service is active (NVSwitch SKUs only)."""
    rr = ctx.run(
        ["systemctl", "is-active", "nvidia-fabricmanager"],
        mock_fixture="fabricmanager_active.txt",
    )
    if rr.skipped:
        return Result(name="fabricmanager", status="SKIP", message="dry-run")
    status = rr.stdout.strip()
    if status == "active":
        return Result(name="fabricmanager", status="PASS", message="active")
    return Result(
        name="fabricmanager",
        status="FAIL",
        message=f"nvidia-fabricmanager is {status!r} (expected active)",
        _raw=rr.stdout + rr.stderr,
    )


@check("numa_topology", category="preflight")
def numa_topology(ctx: RunContext) -> Result:
    """Verify at least 2 NUMA nodes with CPU + memory are present."""
    rr = ctx.run(["numactl", "--hardware"], mock_fixture=f"numactl_{ctx.sku}.txt")
    if rr.skipped:
        return Result(name="numa_topology", status="SKIP", message="dry-run")
    if rr.returncode != 0:
        return Result(name="numa_topology", status="FAIL",
                      message=f"numactl exited {rr.returncode}", _raw=rr.stderr)
    nodes = parsers.parse_numactl_hardware(rr.stdout)
    cpu_nodes = [n for n in nodes if n["cpus"] and n["mem_mb"] > 0]
    metrics = {"numa_nodes_total": len(nodes), "numa_nodes_with_cpu": len(cpu_nodes)}
    if len(cpu_nodes) < 2:
        return Result(name="numa_topology", status="WARN",
                      message=f"only {len(cpu_nodes)} NUMA node(s) with CPU+memory",
                      metrics=metrics)
    return Result(name="numa_topology", status="PASS",
                  message=f"{len(cpu_nodes)} NUMA nodes with CPU+memory",
                  metrics=metrics)


@check("rdma_link", category="preflight")
def rdma_link(ctx: RunContext) -> Result:
    """
    Verify RDMA HCAs are ACTIVE + LINK_UP.

    For A10: discovers active HCAs dynamically (no fixed list).
    For GB300/B300: compares against the expected list in config.yaml.
    """
    net_cfg = ctx.network_config()

    fixture = f"rdma_link_{ctx.sku}.txt"
    rr = ctx.run(["rdma", "link"], mock_fixture=fixture)
    if rr.skipped:
        return Result(name="rdma_link", status="SKIP", message="dry-run")
    if rr.returncode != 0:
        return Result(name="rdma_link", status="FAIL",
                      message=f"rdma link exited {rr.returncode}", _raw=rr.stderr)

    active = parsers.parse_rdma_link(rr.stdout)
    metrics = {"active_hca_count": len(active), "active_hcas": [h["hca"] for h in active]}

    expected_count = net_cfg.get("expected_active_hcas")
    configured_names = net_cfg.get("hca_names", [])

    issues = []

    if expected_count and len(active) < expected_count:
        issues.append(
            f"found {len(active)} active HCA(s), expected {expected_count}"
        )

    if configured_names:
        active_names = {h["hca"] for h in active}
        missing = [n for n in configured_names if n not in active_names]
        if missing:
            issues.append(f"HCAs not active: {missing}")

    if issues:
        return Result(name="rdma_link", status="FAIL",
                      message="; ".join(issues), metrics=metrics, _raw=rr.stdout)

    return Result(name="rdma_link", status="PASS",
                  message=f"{len(active)} HCA(s) ACTIVE+LINK_UP",
                  metrics=metrics)


# ---------------------------------------------------------------------------
# bandwidth
# ---------------------------------------------------------------------------

@check("nvbandwidth", category="bandwidth")
def nvbandwidth(ctx: RunContext) -> Result:
    """
    Run nvbandwidth memory bandwidth tests.

    Test names and thresholds come from config.yaml per SKU.
    For A10: PCIe P2P; for GB300/B300: NVLink.
    """
    sku_cfg = ctx.sku_config()
    nb_cfg = ctx.config.get("nvbandwidth", {})
    sku_nb = nb_cfg.get("tests", {}).get(ctx.sku, {})

    # Resolve test names and threshold
    if isinstance(sku_nb, dict):
        test_names = sku_nb.get("test_names", [])
        # Per-test thresholds dict; fall back to single legacy key
        thresholds: dict[str, float] = sku_nb.get("thresholds", {})
        if not thresholds:
            legacy = sku_nb.get("d2d_read_ce_min_gbs", 0)
            if legacy:
                thresholds["device_to_device_bidirectional_memcpy_read_ce"] = legacy
    else:
        # gb300/b300: list of test names, single threshold in sku config
        test_names = sku_nb if isinstance(sku_nb, list) else []
        legacy = sku_cfg.get("nvbandwidth_d2d_read_ce_min_gbs", 0)
        thresholds = {}
        if legacy:
            thresholds["device_to_device_bidirectional_memcpy_read_ce"] = legacy

    if not test_names:
        return Result(name="nvbandwidth", status="SKIP",
                      message=f"no nvbandwidth tests configured for SKU {ctx.sku!r}")

    tool = ctx.config.get("tools", {}).get("nvbandwidth", "./nvbandwidth")
    fixture = f"nvbandwidth_{ctx.sku}.txt"

    metrics: dict = {}
    issues = []

    for test in test_names:
        rr = ctx.run([tool, "-t", test], mock_fixture=fixture)
        if rr.skipped:
            return Result(name="nvbandwidth", status="SKIP", message="dry-run")
        if rr.returncode != 0:
            return Result(name="nvbandwidth", status="FAIL",
                          message=f"nvbandwidth exited {rr.returncode}: {rr.stderr.strip()[:200]}",
                          _raw=rr.stdout + rr.stderr)

        result = parsers.parse_nvbandwidth(rr.stdout, test)
        gbs = result.get("sum_gbs")
        metrics[f"{test}_sum_gbs"] = gbs

        floor = thresholds.get(test, 0)
        if floor and gbs is not None and gbs < floor:
            issues.append(f"{test}: {gbs:.1f} GB/s < {floor} GB/s")

    if issues:
        return Result(name="nvbandwidth", status="FAIL",
                      message="; ".join(issues), metrics=metrics)

    return Result(
        name="nvbandwidth",
        status="PASS",
        message="; ".join(
            f"{k}={v:.1f} GB/s" for k, v in metrics.items() if v is not None
        ),
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _version_lt(v1: str, v2: str) -> bool:
    """True if v1 < v2 by major version number only."""
    def major(v: str) -> int:
        try:
            return int(v.split(".")[0])
        except (ValueError, IndexError):
            return 0
    return major(v1) < major(v2)
