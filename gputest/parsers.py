"""Pure parsing functions: text in, dict out. No I/O, no side effects."""

from __future__ import annotations

import json
import math
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
    """Read SUM <test> <value> (upstream) or SUM <value> GB/s."""
    rows, total = [], None
    active = False
    for line in text.splitlines():
        if line.startswith('Running '):
            # Older captured output ends in " benchmark".
            active = line in (f'Running {test_name}.', f'Running {test_name} benchmark')
        match = re.match(r'SUM\s+(?:(\S+)\s+)?([\d.eE+-]+)(?:\s+GB/s)?\s*$', line)
        if match and (match[1] == test_name or (match[1] is None and active)):
            total = float(match[2])
        if active:
            row = re.match(r'(.+):\s+([\d.]+)\s+GB/s', line)
            if row:
                rows.append({'label': row[1], 'gbs': float(row[2])})
    return {'sum_gbs': total, 'rows': rows}


def parse_gpu_burn(text: str) -> dict:
    """Parse wilicc progress vectors and gpu-fryer summary lines.

    Sources: wilicc/gpu-burn gpu_burn-drv.cpp; huggingface/gpu-fryer README.
    Error counters are cumulative; preserve the maximum, not their sum over time.
    """
    gpus, faulty = {}, False

    def record(gid, rate, errors=0, temp=None):
        gpu = gpus.setdefault(gid, {'gpu_id': gid, 'gflops': rate, 'errors': 0})
        gpu['gflops'] = rate
        gpu['errors'] = max(gpu['errors'], errors)
        if temp is not None:
            gpu['temp_c'] = max(gpu.get('temp_c', temp), temp)

    current = None
    for line in text.replace('\r', '\n').splitlines():
        if 'FAULTY' in line or 'DIED!' in line:
            faulty = True
        if "proc'd:" in line and 'errors:' in line:
            rates = re.findall(r'\(([\d.]+) Gflop/s\)', line, re.IGNORECASE)
            errors = re.findall(r'\d+', line.split('errors:', 1)[1].split('temps:')[0])
            temps = re.findall(r'(\d+) C|(--)', line.split('temps:')[-1])
            for i, rate in enumerate(rates):
                temp = float(temps[i][0]) if i < len(temps) and temps[i][0] else None
                record(i, float(rate), int(errors[i]) if i < len(errors) else 0, temp)
        m = re.search(r'GPU\s*#?(\d+):\s*([\d.]+)\s+Gflops?/s', line, re.IGNORECASE)
        if m:
            current = int(m[1])
            err = re.search(r'(\d+) errors', line)
            temp = re.search(r'temp\s+([\d.]+)', line)
            record(current, float(m[2]), int(err[1]) if err else 0,
                   float(temp[1]) if temp else None)
        if current is not None and 'Temperature:' in line:
            m = re.search(r'max:\s*([\d.]+)', line)
            if m:
                record(current, gpus[current]['gflops'], temp=float(m[1]))
    count = sum(g['errors'] for g in gpus.values())
    return {'per_gpu': sorted(gpus.values(), key=lambda g: g['gpu_id']),
            'errors': bool(count or faulty), 'error_count': count}


def parse_gpu_samples(text: str) -> dict:
    """nvidia-smi index,temperature.gpu,power.draw CSV; unavailable fields stay absent."""
    peaks = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        gpu = peaks.setdefault(parts[0], {})
        for key, value in zip(('temp_c', 'power_w'), parts[1:]):
            try:
                number = float(value)
            except ValueError:
                continue
            if math.isfinite(number):
                gpu[key] = max(gpu.get(key, number), number)
    return peaks


def parse_babelstream(text: str) -> dict:
    """Read native BabelStream rows; normalize default MB/s to decimal GB/s."""
    benchmarks, scale = {}, 1.0
    for line in text.splitlines():
        if 'MB/s' in line:
            scale = 0.001
        elif 'GB/s' in line:
            scale = 1.0
        m = re.match(r'\s*(Copy|Mul|Add|Triad|Dot):?\s+([\d.eE+-]+)', line)
        if m:
            benchmarks.setdefault(m[1].lower(), []).append(float(m[2]) * scale)
    return {'benchmarks': benchmarks}


def parse_dcgmi_json(text: str) -> dict:
    """Read NVIDIA's test_categories/tests/results JSON and legacy GPU captures.

    Schema: https://github.com/NVIDIA/DCGM/blob/master/testing/python3/dcgm_diag_schema.json
    """
    tests = []
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise TypeError('expected a JSON object')
        diag = data.get('DCGM Diagnostic', data.get('DCGM GPU Diagnostic', {}))
        runtime_error = diag.get('runtime_error')
        for category in diag.get('test_categories', []):
            for test in category.get('tests', []):
                results = list(test.get('results', []))
                statuses = [str(r.get('status', 'UNKNOWN')).upper() for r in results]
                if test.get('test_summary'):
                    statuses.append(str(test['test_summary'].get('status', 'UNKNOWN')).upper())
                normalized = [{'WARNING': 'WARN', 'SKIPPED': 'SKIP'}.get(s, s)
                              for s in statuses]
                if 'FAIL' in normalized or not normalized:
                    outcome = 'FAIL'
                elif any(s not in ('PASS', 'SKIP') for s in normalized):
                    outcome = 'WARN'
                elif all(s == 'SKIP' for s in normalized):
                    outcome = 'SKIP'
                else:
                    outcome = 'PASS'
                tests.append({'name': test['name'], 'result': outcome, 'results': results})
        for gpu in data.get('gpus', []):
            for test in gpu.get('tests', []):
                tests.append({'name': test['name'], 'gpu_id': gpu.get('gpu_id'),
                              'result': str(test.get('result', 'UNKNOWN')).upper()})
        for test in tests:
            test['result'] = {'WARNING': 'WARN', 'SKIPPED': 'SKIP'}.get(
                test['result'], test['result'])
        if runtime_error or not tests or any(t['result'] == 'FAIL' for t in tests):
            status = 'FAIL'
        elif any(t['result'] not in ('PASS', 'SKIP') for t in tests):
            status = 'WARN'
        elif all(t['result'] == 'SKIP' for t in tests):
            status = 'SKIP'
        else:
            status = 'PASS'
        return {'tests': tests, 'status': status, 'runtime_error': runtime_error}
    except (ValueError, KeyError, TypeError, AttributeError):
        return {'tests': [], 'status': 'FAIL', 'runtime_error': 'invalid diagnostic JSON'}


def parse_perftest(text: str, kind: str) -> dict:
    """perftest BW average (Gb/sec) or latency average (usec), not peak/min.

    https://github.com/linux-rdma/perftest (default report columns)
    """
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < (5 if kind == 'bw' else 9) or not fields[0].isdigit():
            continue
        try:
            value = float(fields[3 if kind == 'bw' else 5])
            if math.isfinite(value) and value > 0:
                rows.append({'size': int(fields[0]), 'value': value})
        except ValueError:
            continue
    return {'rows': rows, 'value': rows[-1]['value'] if rows else None}


def parse_nccl(text: str) -> dict:
    """nccl-tests performance.cu table: retain both placement variants.

    https://github.com/NVIDIA/nccl-tests/blob/master/src/common.cu
    """
    rows = []
    for line in text.splitlines():
        f = line.split()
        if len(f) != 13 or not f[0].isdigit():
            continue
        try:
            row = {'size': int(f[0])}
            for label, offset in (('out_of_place', 5), ('in_place', 9)):
                values = [float(v) for v in f[offset:offset + 3]]
                if not all(math.isfinite(v) and v >= 0 for v in values):
                    raise ValueError('invalid performance value')
                row[label] = dict(zip(('time_us', 'algbw', 'busbw'), values))
                row[label]['wrong'] = None if f[offset + 3] == 'N/A' else int(f[offset + 3])
            rows.append(row)
        except ValueError:
            continue
    return {'rows': rows, 'largest': max(rows, key=lambda r: r['size']) if rows else None}


def parse_ip_addresses(text: str) -> dict:
    """ip -j address output, indexed by interface; exclude tentative addresses."""
    result = {}
    for interface in json.loads(text):
        result[interface['ifname']] = [a for a in interface.get('addr_info', [])
                                       if not a.get('tentative') and not a.get('dadfailed')]
    return result


def parse_gpu_topology(text: str) -> dict:
    """Read the nvidia-smi topo -m matrix header, excluding NIC legend text."""
    for line in text.splitlines():
        fields = line.split()
        if fields and re.fullmatch(r'GPU\d+', fields[0]):
            gpus = [f for f in fields if re.fullmatch(r'GPU\d+', f)]
            nics = [f for f in fields if re.fullmatch(r'NIC\d+', f)]
            if len(gpus) > 1 or 'CPU' in fields:
                return {'gpus': gpus, 'nics': nics}
    return {'gpus': [], 'nics': []}
