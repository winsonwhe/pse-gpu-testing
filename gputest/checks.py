"""All check functions. Import side-effect: registers checks in core._REGISTRY."""

from __future__ import annotations

import math
import re
import statistics

from gputest import multihost_checks, parsers
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
        [ctx.config["tools"]["nvidia_smi"],
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
        if gpu_count_min is not None and (actual_count < gpu_count_min or actual_count > expected_count):
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
        [ctx.config["tools"]["nvidia_smi"], "--query-gpu=compute_cap", "--format=csv,noheader"],
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
            if mem is None:
                issues.append(f"GPU {g['index']}: memory unavailable")
            elif abs(mem - mem_expected) > sku["memory_tolerance_mib"]:
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
    """Check controller->targets and launcher->peers, in parallel through run()."""
    tools = ctx.config['tools']
    jobs = [{'cmd': [tools['hostname']], 'node': n, 'mock_fixture': 'ssh_hostname.txt',
             'timeout': ctx.config['executor']['timeout_s']} for n in ctx.nodes]
    jobs += [{'cmd': [tools['ssh'], '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '--', n,
                      tools['hostname']], 'node': ctx.nodes[0], 'mock_fixture': 'ssh_hostname.txt',
              'timeout': ctx.config['executor']['timeout_s']} for n in ctx.nodes[1:]]
    results = ctx.run_parallel(jobs)
    failed = [job['node'] for job, rr in zip(jobs, results) if rr.returncode]
    raw = '\n'.join(f'{job["node"]}: {rr.stdout}{rr.stderr}' for job, rr in zip(jobs, results))
    return Result('ssh_connectivity', 'SKIP' if ctx.dry_run else ('FAIL' if failed else 'PASS'),
                  'dry-run' if ctx.dry_run else (f'SSH broken: {failed}' if failed else 'controller and launcher SSH OK'),
                  {'nodes_checked': len(ctx.nodes), 'failed_links': failed}, _raw=raw)


@check("fabricmanager", category="preflight", skus=["gb300", "b300"])
def fabricmanager(ctx: RunContext) -> Result:
    """Check nvidia-fabricmanager service is active (NVSwitch SKUs only)."""
    if not ctx.sku_config().get("fabricmanager_required", False):
        return Result("fabricmanager", "SKIP", "not applicable: Fabric Manager is not required")
    rr = ctx.run(
        [ctx.config["tools"]["systemctl"], "is-active", "nvidia-fabricmanager"],
        mock_fixture="fabricmanager_active.txt",
    )
    if rr.skipped:
        return Result(name="fabricmanager", status="SKIP", message="dry-run")
    status = rr.stdout.strip()
    if status == "active" and rr.returncode == 0:
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
    rr = ctx.run([ctx.config["tools"]["numactl"], "--hardware"], mock_fixture=f"numactl_{ctx.sku}.txt")
    if rr.skipped:
        return Result(name="numa_topology", status="SKIP", message="dry-run")
    if rr.returncode != 0:
        return Result(name="numa_topology", status="FAIL",
                      message=f"numactl exited {rr.returncode}", _raw=rr.stderr)
    nodes = parsers.parse_numactl_hardware(rr.stdout)
    cpu_nodes = [n for n in nodes if n["cpus"] and n["mem_mb"] > 0]
    metrics = {"numa_nodes_total": len(nodes), "numa_nodes_with_cpu": len(cpu_nodes)}
    if len(cpu_nodes) < ctx.sku_config()["numa_cpu_nodes_min"]:
        return Result(name="numa_topology", status="WARN",
                      message=f"only {len(cpu_nodes)} NUMA node(s) with CPU+memory",
                      metrics=metrics, _raw=rr.stdout)
    return Result(name="numa_topology", status="PASS",
                  message=f"{len(cpu_nodes)} NUMA nodes with CPU+memory",
                  metrics=metrics, _raw=rr.stdout)


@check("rdma_link", category="preflight")
def rdma_link(ctx: RunContext) -> Result:
    """Compatibility alias for the shared device/VRF discovery check."""
    result = multihost_checks.rdma_iface(ctx)
    result.name = 'rdma_link'
    devices = next(iter(result.metrics['devices'].values()), [])
    result.metrics.update(active_hca_count=len(devices), active_hcas=[d['hca'] for d in devices])
    return result


@check('gpu_topology', category='preflight')
def gpu_topology(ctx: RunContext) -> Result:
    rr = ctx.run([ctx.config['tools']['nvidia_smi'], 'topo', '-m'],
                 mock_fixture=f'gpu_topology_{ctx.sku}.txt')
    parsed = parsers.parse_gpu_topology(rr.stdout)
    net = ctx.network_config()
    expected_nics = len(net['vf_rails']) if ctx.network_mode == 'multiplanar' else net['expected_active_hcas']
    issues = []
    if rr.returncode:
        issues.append(f'topology command exited {rr.returncode}')
    if len(parsed['gpus']) != ctx.sku_config()['gpu_count']:
        issues.append('missing or unexpected GPU in topology output')
    if len(parsed['nics']) < expected_nics:
        issues.append('missing NIC in topology output')
    return Result('gpu_topology', 'SKIP' if ctx.dry_run else ('FAIL' if issues else 'PASS'),
                  'dry-run' if ctx.dry_run else ('; '.join(issues) or 'GPU/NIC topology complete'),
                  parsed, _raw=rr.stdout + rr.stderr)


def _diagnostic_result(ctx, name, cmd, fixture, parser=None, *, warn_on=None):
    """Run a read-only diagnostic and normalize unavailable tools to SKIP."""
    rr = ctx.run(cmd, timeout=ctx.config.get('diagnostics', {}).get('timeout_s', 30),
                 mock_fixture=fixture)
    if rr.skipped:
        return Result(name, 'SKIP', 'dry-run')
    if rr.returncode == 127:
        return Result(name, 'SKIP', f'tool unavailable: {rr.stderr.strip()}', _raw=rr.stderr)
    if rr.returncode:
        return Result(name, 'FAIL', f'exited {rr.returncode}: {rr.stderr.strip()[-300:]}',
                      _raw=rr.stdout + rr.stderr)
    metrics = parser(rr.stdout) if parser else {}
    bad = warn_on(metrics) if warn_on else False
    return Result(name, 'WARN' if bad else 'PASS',
                  'diagnostic findings detected' if bad else 'no diagnostic findings',
                  metrics, _raw=rr.stdout + rr.stderr)


@check('kernel_xid', category='preflight')
def kernel_xid(ctx: RunContext) -> Result:
    """Scan kernel logs for NVIDIA Xid events; codes are retained verbatim."""
    cmd = [ctx.config['tools'].get('journalctl', 'journalctl'), '-k', '--no-pager', '-b']
    result = _diagnostic_result(ctx, 'kernel_xid', cmd, 'kernel_log.txt', parsers.parse_xid_events,
                                warn_on=lambda m: bool(m))
    events = result.metrics if isinstance(result.metrics, list) else []
    result.metrics = {'events': events, 'codes': [e['code'] for e in events],
                      'event_count': len(events)}
    if events and result.status == 'WARN':
        result.message = 'NVIDIA Xid events: ' + ', '.join(str(e['code']) for e in events)
    return result


@check('ecc_remap', category='preflight')
def ecc_remap(ctx: RunContext) -> Result:
    """Check ECC counters, retired pages and row-remapper pending/failure state."""
    cmd = [ctx.config['tools']['nvidia_smi'], '-q', '-d', 'ECC,REMAP,RETIRED_PAGES']
    rr = ctx.run(cmd, mock_fixture='nvidia_smi_health.txt')
    if rr.skipped:
        return Result('ecc_remap', 'SKIP', 'dry-run')
    if rr.returncode == 127:
        return Result('ecc_remap', 'SKIP', 'nvidia-smi unavailable', _raw=rr.stderr)
    fields = parsers.parse_nvidia_smi_key_value(rr.stdout)
    bad_fields, warning_fields = {}, {}
    for key, values in fields.items():
        low = ' '.join(values).lower()
        if any(token in key for token in ('uncorrectable', 'retired', 'pending', 'failure', 'correctable')):
            numbers = [int(x) for x in re.findall(r'\d+', low)]
            finding = (numbers and max(numbers) > 0) or any(x in low for x in ('pending', 'failure', 'yes'))
            if finding:
                (warning_fields if 'correctable' in key and 'uncorrectable' not in key else bad_fields)[key] = values
    status = 'FAIL' if bad_fields else ('WARN' if warning_fields else 'PASS')
    message = ('ECC/row-remap failures detected' if bad_fields else
               'correctable ECC events detected' if warning_fields else 'ECC and row-remap healthy')
    return Result('ecc_remap', status, message,
                  {'fields': fields, 'findings': bad_fields, 'warnings': warning_fields},
                  _raw=rr.stdout + rr.stderr)


@check('pcie_link', category='preflight')
def pcie_link(ctx: RunContext) -> Result:
    """Report current PCIe generation/width and AER counters."""
    cmd = [ctx.config['tools']['nvidia_smi'], '-q', '-d', 'PCI']
    return _diagnostic_result(ctx, 'pcie_link', cmd, 'nvidia_smi_pcie.txt', parsers.parse_pcie_link,
                              warn_on=lambda m: any(v > 0 for v in m.get('aer', {}).values()))


@check('nvlink', category='preflight', skus=['gb300', 'b300'])
def nvlink(ctx: RunContext) -> Result:
    """Check NVLink link state and error counters."""
    return _diagnostic_result(ctx, 'nvlink', [ctx.config['tools']['nvidia_smi'], 'nvlink', '-s'],
                              'nvidia_smi_nvlink.txt', parsers.parse_nvlink_status,
                              warn_on=lambda m: any(v > 0 for v in m.get('errors', {}).values()))


@check('clocks_throttle', category='preflight')
def clocks_throttle(ctx: RunContext) -> Result:
    """Capture NVIDIA clock throttle reasons for later comparison with burn."""
    cmd = [ctx.config['tools']['nvidia_smi'],
           '--query-gpu=index,clocks_throttle_reasons.active', '--format=csv,noheader']
    return _diagnostic_result(ctx, 'clocks_throttle', cmd, 'nvidia_smi_throttle.txt',
                              lambda text: {'rows': [line.strip() for line in text.splitlines() if line.strip()]},
                              warn_on=lambda m: any('none' not in x.lower() and 'no throttle' not in x.lower()
                                                    for x in m['rows']))


@check('software_versions', category='preflight')
def software_versions(ctx: RunContext) -> Result:
    """Collect driver/CUDA/VBIOS/NIC/OFED/DCGM/Fabric Manager versions."""
    commands = [
        ([ctx.config['tools']['nvidia_smi'], '--query-gpu=driver_version,vbios_version', '--format=csv,noheader'], 'nvidia_smi_health.txt'),
        ([ctx.config['tools'].get('nvcc', 'nvcc'), '--version'], 'version_nvcc.txt'),
        ([ctx.config['tools'].get('ofed_info', 'ofed_info'), '-s'], 'version_ofed.txt'),
        ([ctx.config['tools']['dcgmi'], '--version'], 'version_dcgmi.txt'),
        ([ctx.config['tools'].get('ibv_devinfo', 'ibv_devinfo'), '--version'], 'version_nic.txt'),
        ([ctx.config['tools'].get('fabricmanager', 'nvidia-fabricmanager'), '--version'], 'version_fabricmanager.txt'),
    ]
    raw, versions, missing = [], {}, []
    for cmd, fixture in commands:
        rr = ctx.run(cmd, mock_fixture=fixture)
        raw.append(rr.stdout + rr.stderr)
        if rr.skipped:
            return Result('software_versions', 'SKIP', 'dry-run')
        if rr.returncode == 127:
            missing.append(cmd[0]); continue
        if rr.returncode == 0:
            versions[cmd[0]] = rr.stdout.strip()
    status = 'WARN' if missing else 'PASS'
    return Result('software_versions', status,
                  ('unavailable: ' + ', '.join(missing)) if missing else 'version inventory collected',
                  {'versions': versions, 'unavailable': missing}, _raw='\n'.join(raw))


@check('gpudirect_rdma', category='rdma')
def gpudirect_rdma(ctx: RunContext) -> Result:
    """Verify nvidia-peermem and that perftest exposes CUDA mode."""
    module = ctx.run([ctx.config['tools'].get('lsmod', 'lsmod')], mock_fixture='lsmod.txt')
    probe = ctx.run([ctx.config['tools']['ib_write_bw'], '--help'], mock_fixture='ib_write_bw_help.txt')
    if module.skipped or probe.skipped:
        return Result('gpudirect_rdma', 'SKIP', 'dry-run')
    loaded = bool(re.search(r'^nvidia_peermem\b', module.stdout, re.M))
    supports_cuda = '--use_cuda' in (probe.stdout + probe.stderr)
    if probe.returncode == 127:
        return Result('gpudirect_rdma', 'SKIP', 'ib_write_bw unavailable', _raw=probe.stderr)
    status = 'PASS' if loaded and supports_cuda else 'FAIL'
    return Result('gpudirect_rdma', status,
                  'nvidia-peermem loaded and ib_write_bw supports --use_cuda' if status == 'PASS'
                  else 'GPUDirect RDMA prerequisites missing',
                  {'nvidia_peermem_loaded': loaded, 'ib_write_bw_use_cuda': supports_cuda},
                  _raw=module.stdout + probe.stdout + module.stderr + probe.stderr)


def _prepare(ctx, name, section, cmake_flags=()):
    """Use an installed binary, or clone/build at the configured source path."""
    import shutil
    from pathlib import Path

    cfg = ctx.config[section]
    tool = ctx.config['tools'][name]
    if ctx.target_node:
        # Remote binaries are provisioned beforehand; do not inspect/build on the controller.
        return tool, None
    installed = shutil.which(tool)
    if not ctx.mock and installed:
        return str(Path(installed).resolve()), None
    if not cfg.get('build_if_missing', False):
        return tool, None
    source = Path(cfg['source_dir']).resolve()
    nvcc = str(Path(ctx.config['cuda']['path']) / 'bin/nvcc')
    tools = ctx.config['tools']
    deps = [tools['git'], tools['make'] if section == 'gpu_burn' else tools['cmake'], nvcc]
    if not ctx.dry_run and not ctx.mock:
        missing = [d for d in deps if not shutil.which(d)]
        if missing:
            return tool, Result(section, 'SKIP', 'missing build tools: ' + ', '.join(missing))
    commands = []
    if not source.exists():
        if not ctx.dry_run and not ctx.mock:
            source.parent.mkdir(parents=True, exist_ok=True)
        commands.append([tools['git'], 'clone', '--depth', '1', cfg['repository'], str(source)])
    if section == 'gpu_burn':
        arch = str(cfg['compute_capability'][ctx.sku]).removeprefix('sm_')
        commands.append([tools['make'], '-C', str(source), f'COMPUTE={arch}',
                         'CUDAPATH=' + ctx.config['cuda']['path']])
    else:
        commands.extend([
            [tools['cmake'], '-S', str(source), '-B', str(source / 'build'),
             '-DCMAKE_CUDA_COMPILER=' + nvcc, *cmake_flags],
            [tools['cmake'], '--build', str(source / 'build'), '-j', str(cfg['build_jobs'])],
        ])
    for cmd in commands:
        rr = ctx.run(cmd, timeout=cfg['build_timeout_s'], mock_fixture='build.txt')
        if rr.returncode:
            return tool, Result(section, 'SKIP' if rr.returncode == 127 else 'FAIL',
                                f'build command failed: {cmd[0]}: {rr.stderr[-300:]}')
    return tool, None


def _command_problem(name, rr):
    if rr.skipped:
        return Result(name, 'SKIP', 'dry-run')
    if rr.returncode == 127:
        return Result(name, 'SKIP', f'tool unavailable: {rr.stderr.strip()}')
    if rr.returncode:
        return Result(name, 'FAIL', f'exited {rr.returncode}: {rr.stderr.strip()[-300:]}',
                      _raw=rr.stdout + rr.stderr)
    return None


@check('gpu_burn', category='burn')
def gpu_burn(ctx: RunContext) -> Result:
    """Stress GPUs, retain errors, compare throughput, and sample temperature/power."""
    from pathlib import Path

    from gputest.core import run_sampled

    cfg = ctx.config['gpu_burn']
    # Xid is cumulative in the kernel log. Capture both sides so a pre-existing
    # event does not get misreported as a burn-induced failure.
    log_cmd = [ctx.config['tools'].get('journalctl', 'journalctl'), '-k', '--no-pager', '-b']
    xid_before = ctx.run(log_cmd, mock_fixture='kernel_log.txt')
    backend = cfg['backend']
    selection = cfg.get('gpus', [])
    env = {'CUDA_VISIBLE_DEVICES': ','.join(map(str, selection))} if selection else None
    cwd = None
    if backend == 'gpu-burn':
        tool, problem = _prepare(ctx, 'gpu-burn', 'gpu_burn')
        if problem:
            return problem
        # gpu-burn loads compare.ptx relative to its working directory.
        cwd = Path(tool).resolve().parent
        cmd = [str(Path(tool).resolve()), '-m', f"{cfg['memory_percentage']}%"]
        if cfg['tensor_core']:
            cmd.append('-tc')
        if cfg['double']:
            cmd.append('-d')
        fixture = 'gpu_burn.txt'
    elif backend == 'gpu-fryer':
        cmd = [ctx.config['tools']['gpu_fryer']]
        if selection:
            # https://github.com/huggingface/gpu-fryer/blob/main/src/main.rs (Args.gpus)
            cmd += ['--gpus', ','.join(map(str, selection))]
        env = None
        nvml = cfg.get('nvml_lib_path') or cfg.get('nvml_lib_' + ctx.sku_config()['arch'])
        if nvml:
            cmd += ['--nvml-lib-path', nvml]
        fixture = 'gpu_fryer.txt'
    else:
        return Result('gpu_burn', 'FAIL', f'unknown backend: {backend}')
    duration = ctx.burn_duration_s()
    if ctx.target_node:
        from pathlib import PurePosixPath
        remote_path = PurePosixPath(tool if backend == 'gpu-burn' else cmd[0])
        if backend == 'gpu-burn' and '/' in str(remote_path):
            cwd = str(remote_path.parent)
            cmd[0] = './' + remote_path.name
        else:
            cwd = None
    cmd.append(str(duration))
    rr, samples, sample_error = run_sampled(
        ctx, cmd, timeout=duration + cfg['timeout_margin_s'],
        fixture=fixture, cwd=cwd, env=env,
    )
    xid_after = ctx.run(log_cmd, mock_fixture='kernel_log.txt')
    throttle_rr = ctx.run(
        [ctx.config['tools']['nvidia_smi'],
         '--query-gpu=index,clocks_throttle_reasons.active', '--format=csv,noheader'],
        mock_fixture='nvidia_smi_throttle.txt',
    )
    problem = _command_problem('gpu_burn', rr)
    if problem and problem.status == 'SKIP':
        return problem
    parsed = parsers.parse_gpu_burn(rr.stdout + '\n' + rr.stderr)
    gpus = parsed['per_gpu']
    # wilicc progress vectors follow CUDA_VISIBLE_DEVICES order.
    if selection and backend == 'gpu-burn':
        for gpu in gpus:
            if gpu['gpu_id'] < len(selection):
                gpu['gpu_id'] = selection[gpu['gpu_id']]
    expected = len(selection) if selection else ctx.sku_config()['gpu_count']
    issues = []
    if problem:
        issues.append(problem.message)
    if len(gpus) != expected:
        issues.append(f'expected {expected} GPU results, parsed {len(gpus)}')
    if parsed['errors']:
        issues.append(f"GPU errors detected (counter: {parsed['error_count']})")
    rates = [g['gflops'] for g in gpus]
    mean = statistics.mean(rates) if rates else 0
    for gpu in gpus:
        deviation = abs(gpu['gflops'] - mean) / mean * 100 if mean > 0 else 100
        if gpu['gflops'] <= 0 or deviation > cfg['perf_deviation_threshold_pct']:
            issues.append(f"GPU {gpu['gpu_id']}: throughput deviation {deviation:.1f}%")
    peaks = parsers.parse_gpu_samples(samples)
    if selection:
        peaks = {k: v for k, v in peaks.items() if int(k) in selection}
    metrics = {'duration_s': duration, 'backend': backend, 'gpu_count': len(gpus), 'per_gpu': gpus,
               'mean_gflops': mean, 'per_gpu_gflops': rates, 'sampled_peaks': peaks,
               'numerical_error_check': backend == 'gpu-burn'}
    before_events = parsers.parse_xid_events(xid_before.stdout)
    after_events = parsers.parse_xid_events(xid_after.stdout)
    before_counts = {}
    for event in before_events:
        before_counts[event['code']] = before_counts.get(event['code'], 0) + 1
    xid_delta = []
    remaining = dict(before_counts)
    for event in after_events:
        if remaining.get(event['code'], 0):
            remaining[event['code']] -= 1
        else:
            xid_delta.append(event)
    metrics['xid_before'] = before_events
    metrics['xid_after'] = after_events
    metrics['xid_delta'] = xid_delta
    if xid_delta:
        issues.append('new kernel Xid event(s): ' + ', '.join(str(e['code']) for e in xid_delta))
    throttle = {}
    for line in throttle_rr.stdout.splitlines():
        fields = [part.strip() for part in line.split(',', 1)]
        if len(fields) == 2 and fields[1] and 'none' not in fields[1].lower() and 'no throttle' not in fields[1].lower():
            throttle[fields[0]] = fields[1]
    if throttle:
        metrics['throttle_reasons'] = throttle
        issues.append('clock throttle during burn: ' + str(throttle))
    for field in ('temp_c', 'power_w'):
        values = [g[field] for g in peaks.values() if field in g]
        if values:
            metrics['peak_' + field] = max(values)
    if not sample_error and any(
        field not in peaks.get(str(gpu['gpu_id']), {})
        for gpu in gpus for field in ('temp_c', 'power_w')
    ):
        sample_error = 'temperature/power samples incomplete for tested GPUs'
    if sample_error:
        metrics['sampling_warning'] = sample_error
    status = 'FAIL' if issues else ('WARN' if sample_error else 'PASS')
    message = '; '.join(issues) or f'{len(gpus)} GPUs, {mean:.0f} mean GFlop/s'
    if sample_error:
        message += '; ' + sample_error
    if backend == 'gpu-fryer':
        message += '; gpu-fryer does not validate numerical errors'
    return Result('gpu_burn', status, message, metrics, _raw=rr.stdout + rr.stderr)


@check('nvbandwidth', category='bandwidth')
def nvbandwidth(ctx: RunContext) -> Result:
    """Measure configured SUM bandwidths; optionally launch documented MPI tests."""
    cfg = ctx.config['nvbandwidth']
    sku = cfg['tests'].get(ctx.sku, {})
    names = sku.get('test_names', [])
    if not names:
        return Result('nvbandwidth', 'SKIP', 'no tests configured for this SKU')
    multi = cfg.get('multinode', False)
    if multi and (ctx.sku != 'gb300' or len(ctx.nodes) < 2):
        return Result('nvbandwidth', 'SKIP', 'multi-node NVLink requires GB300 and >=2 hosts')
    if any(n.startswith('multinode_') != multi for n in names):
        return Result('nvbandwidth', 'FAIL', 'test names must match multinode setting')
    if multi:
        placed = multihost_checks.placement(ctx)
        inventory = placed.metrics['inventory']
        if not ctx.dry_run and len({entry['gmf'] for entry in inventory.values()}) != 1:
            return Result('nvbandwidth', 'FAIL', 'multi-node NVLink must stay inside one GMF')
        imex = multihost_checks.imex(ctx, inventory=inventory)
        if imex.status == 'FAIL':
            return Result('nvbandwidth', 'FAIL', imex.message, _raw=imex._raw)
    flags = ['-DCMAKE_CUDA_ARCHITECTURES=' + str(cfg['cuda_arch'][ctx.sku]),
             '-DMULTINODE=' + ('1' if multi else '0')]
    tool, problem = _prepare(ctx, 'nvbandwidth', 'nvbandwidth', flags)
    if problem:
        return problem
    metrics, issues, raw = {}, [], []
    uncalibrated = False
    for name in names:
        cmd = [tool, '-t', name]
        if multi:
            # https://github.com/NVIDIA/nvbandwidth#multi-node-support
            # One MPI rank per GPU; IMEX must already be configured on every host.
            ranks = ctx.sku_config()['gpu_count']
            cmd = [ctx.config['tools']['mpirun'], '--host', ','.join(ctx.nodes),
                   '--map-by', f'ppr:{ranks}:node', '-np', str(ranks * len(ctx.nodes)),
                   *cfg['mpi_args'], *cmd]
        rr = ctx.run(cmd, timeout=cfg['timeout_s'], mock_fixture=f'nvbandwidth_{ctx.sku}.txt')
        problem = _command_problem('nvbandwidth', rr)
        if problem and not ctx.dry_run:
            return problem
        if ctx.dry_run:
            continue
        raw.append(rr.stdout + rr.stderr)
        value = parsers.parse_nvbandwidth(rr.stdout, name)['sum_gbs']
        metrics[name + '_sum_gbs'] = value
        if name not in sku.get('thresholds', {}):
            uncalibrated = True
        if value is None:
            issues.append(f'{name}: no SUM parsed')
        elif not math.isfinite(value) or value <= 0 or value < sku.get('thresholds', {}).get(name, 0):
            issues.append(f'{name}: {value:.1f} GB/s below configured minimum')
    metrics['uncalibrated'] = uncalibrated
    return Result('nvbandwidth', 'SKIP' if ctx.dry_run else ('FAIL' if issues else ('WARN' if uncalibrated else 'PASS')),
                  'dry-run' if ctx.dry_run else ('; '.join(issues) or
                  ('some SUM thresholds unset' if uncalibrated else 'all SUM thresholds met')),
                  metrics, _raw='\n'.join(raw))


@check('babelstream', category='bandwidth')
def babelstream(ctx: RunContext) -> Result:
    """Run CUDA BabelStream separately for each selected GPU."""
    cfg = ctx.config['babelstream']
    # https://github.com/UoB-HPC/BabelStream/blob/main/src/cuda/model.cmake
    flags = ['-DMODEL=cuda', '-DCUDA_ARCH=' + cfg['cuda_arch'][ctx.sku]]
    tool, problem = _prepare(ctx, 'babelstream', 'babelstream', flags)
    if problem:
        return problem
    devices = cfg.get('gpus') or list(range(ctx.sku_config()['gpu_count']))
    per_gpu, issues, raw = {}, [], []
    uncalibrated = False
    for gpu in devices:
        rr = ctx.run([tool, '--device', str(gpu)], timeout=cfg['timeout_s'],
                     mock_fixture=f'babelstream_{ctx.sku}.txt')
        problem = _command_problem('babelstream', rr)
        if problem and not ctx.dry_run:
            return problem
        if ctx.dry_run:
            continue
        raw.append(f'GPU {gpu}\n' + rr.stdout + rr.stderr)
        benches = parsers.parse_babelstream(rr.stdout)['benchmarks']
        per_gpu[str(gpu)] = benches
        for name in ('copy', 'mul', 'add', 'triad', 'dot'):
            if name + '_min_gbs' not in cfg['thresholds'].get(ctx.sku, {}):
                uncalibrated = True
            values = benches.get(name, [])
            if len(values) != 1:
                issues.append(f'GPU {gpu} {name}: missing or ambiguous benchmark')
            elif (not math.isfinite(values[0]) or
                  values[0] <= 0 or values[0] < cfg['thresholds'].get(ctx.sku, {}).get(name + '_min_gbs', 0)):
                issues.append(f'GPU {gpu} {name}: {values[0]:.1f} GB/s below minimum')
    metrics = {'gpu_count': len(per_gpu), 'per_gpu': per_gpu, 'uncalibrated': uncalibrated}
    for name in ('copy', 'mul', 'add', 'triad', 'dot'):
        values = [b[name][0] for b in per_gpu.values() if b.get(name)]
        if values:
            mean = statistics.mean(values)
            metrics[name + '_mean_gbs'] = mean
            metrics[name + '_spread_gbs'] = max(values) - min(values)
            metrics[name + '_spread_pct'] = (max(values) - min(values)) / mean * 100 if mean else 0
    return Result('babelstream', 'SKIP' if ctx.dry_run else ('FAIL' if issues else ('WARN' if uncalibrated else 'PASS')),
                  'dry-run' if ctx.dry_run else ('; '.join(issues) or f'{len(devices)} GPUs benchmarked' + ('; some thresholds unset' if uncalibrated else '')),
                  metrics, _raw='\n'.join(raw))


@check('dcgmi', category='diagnostics')
def dcgmi(ctx: RunContext) -> Result:
    """Run profile-selected DCGM JSON diagnostics, retaining individual results."""
    cfg = ctx.config['dcgmi']
    profile = ctx.config.get('_profile', 'standard')
    level = cfg['profile_levels'][profile]
    rr = ctx.run([ctx.config['tools']['dcgmi'], 'diag', '-r', str(level), '-j'],
                 timeout=cfg['timeout_s'], mock_fixture=f'dcgmi_r{level}.txt')
    problem = _command_problem('dcgmi', rr)
    if problem and problem.status == 'SKIP':
        return problem
    parsed = parsers.parse_dcgmi_json(rr.stdout)
    tests = parsed['tests']
    metrics = {'level': level, 'tests': tests, 'test_count': len(tests)}
    metrics.update({s.lower(): sum(t['result'] == s for t in tests)
                    for s in ('PASS', 'WARN', 'FAIL', 'SKIP')})
    status = 'FAIL' if problem else parsed['status']
    message = (problem.message if problem else parsed.get('runtime_error')) or (
        f"level {level}: {metrics['pass']} passed, {metrics['warn']} warned, "
        f"{metrics['fail']} failed, {metrics['skip']} skipped")
    return Result('dcgmi', status, message, metrics, _raw=rr.stdout + rr.stderr)


def _version_lt(v1: str, v2: str) -> bool:
    import re
    from itertools import zip_longest

    versions = list(zip_longest(re.findall(r'\d+', v1), re.findall(r'\d+', v2), fillvalue='0'))
    return tuple(int(a) for a, _ in versions) < tuple(int(b) for _, b in versions)
