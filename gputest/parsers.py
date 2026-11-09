"""Pure parsing functions: text in, dict out. No I/O, no side effects."""

from __future__ import annotations

import re


def parse_nvidia_smi_csv(text: str) -> list[dict]:
    """
    Parse output of:
      nvidia-smi --query-gpu=index,name,driver_version,memory.total,ecc.mode.current
                 --format=csv,noheader,nounits

    Returns a list of dicts, one per GPU. Memory total is in MiB (integer).
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            mem_mib = int(parts[3])
        except ValueError:
            mem_mib = None
        rows.append({
            "index": parts[0],
            "name": parts[1],
            "driver_version": parts[2],
            "memory_total_mib": mem_mib,
            "ecc_mode": parts[4],
        })
    return rows


def parse_rdma_link(text: str) -> list[dict]:
    """
    Parse output of `rdma link`.

    Each active line looks like:
      link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev rdma0

    Returns only links where state==ACTIVE and physical_state==LINK_UP.
    """
    active = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.search(r"link\s+(\S+)\s+state\s+(\S+)\s+physical_state\s+(\S+)", line)
        if not m:
            continue
        hca_port, state, phys_state = m.group(1), m.group(2), m.group(3)
        hca = hca_port.split("/")[0]
        if state == "ACTIVE" and phys_state == "LINK_UP":
            netdev_m = re.search(r"netdev\s+(\S+)", line)
            active.append({
                "hca": hca,
                "port": hca_port,
                "state": state,
                "physical_state": phys_state,
                "netdev": netdev_m.group(1) if netdev_m else None,
            })
    return active


def parse_numactl_hardware(text: str) -> list[dict]:
    """
    Parse output of `numactl --hardware`.

    Returns a list of dicts, one per NUMA node:
      {"node": int, "cpus": list[int], "mem_mb": int}

    Only nodes with at least one CPU *and* non-zero memory are considered
    "real" compute nodes; zero-memory nodes (GPU NUMA nodes) are returned
    with mem_mb=0.
    """
    nodes: dict[int, dict] = {}

    for line in text.splitlines():
        line = line.strip()

        # "node 0 cpus: 0 1 2 ..."
        m = re.match(r"node\s+(\d+)\s+cpus:\s*(.*)", line)
        if m:
            nid = int(m.group(1))
            cpus = [int(x) for x in m.group(2).split() if x.isdigit()]
            nodes.setdefault(nid, {"node": nid, "cpus": [], "mem_mb": 0})
            nodes[nid]["cpus"] = cpus
            continue

        # "node 0 size: 979968 MB"
        m = re.match(r"node\s+(\d+)\s+size:\s*(\d+)\s+MB", line)
        if m:
            nid = int(m.group(1))
            nodes.setdefault(nid, {"node": nid, "cpus": [], "mem_mb": 0})
            nodes[nid]["mem_mb"] = int(m.group(2))

    return sorted(nodes.values(), key=lambda n: n["node"])


def parse_nvbandwidth(text: str, test_name: str) -> dict:
    """
    Parse output of `nvbandwidth -t <test_name>`.

    Returns {"sum_gbs": float | None, "rows": [{"label": str, "gbs": float}]}.
    Looks for lines matching the test section and a "SUM" line.
    """
    rows = []
    sum_gbs = None
    in_section = False

    for line in text.splitlines():
        line = line.strip()
        if test_name in line:
            in_section = True
            continue
        if not in_section:
            continue
        # SUM line: "SUM 24.51 GB/s"
        m = re.match(r"SUM\s+([\d.]+)\s+GB/s", line)
        if m:
            sum_gbs = float(m.group(1))
            break
        # Individual line: "memcpy CE GPU 0 -> GPU 1: 24.53 GB/s"
        m2 = re.match(r"(.+):\s+([\d.]+)\s+GB/s", line)
        if m2:
            rows.append({"label": m2.group(1).strip(), "gbs": float(m2.group(2))})

    return {"sum_gbs": sum_gbs, "rows": rows}
