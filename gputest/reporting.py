"""Reports, threshold provenance and known-good performance snapshots."""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import sys
import textwrap
from pathlib import Path


def annotate(result, ctx):
    """Keep provenance next to the limits actually configured for each check."""
    cfg, name = ctx.config, result.name
    thresholds = result.thresholds

    def add(metric, value, rule, path, source='config default'):
        if value is not None:
            thresholds.append({'metric': metric, 'value': value, 'rule': rule,
                               'source': source, 'reference': path})

    if name == 'nvidia_smi':
        for metric, rule in [('gpu_count', 'expected'), ('gpu_count_min', '>='),
                             ('driver_min', '>='), ('mem_mib_per_gpu', 'expected'),
                             ('memory_tolerance_mib', '+/-')]:
            add(metric, ctx.sku_config().get(metric), rule, f'skus.{ctx.sku}.{metric}')
    elif name == 'numa_topology':
        add('CPU + memory NUMA nodes', ctx.sku_config()['numa_cpu_nodes_min'], '>=',
            f'skus.{ctx.sku}.numa_cpu_nodes_min')
    elif name in ('rdma_iface', 'rdma_link', 'gpu_topology'):
        net = ctx.network_config()
        count = len(net['vf_rails']) if ctx.network_mode == 'multiplanar' else net['expected_active_hcas']
        add('active RDMA devices / topology NICs', count, '>=', f'network.{ctx.network_mode}')
        if name == 'gpu_topology':
            add('topology GPUs', ctx.sku_config()['gpu_count'], '==', f'skus.{ctx.sku}.gpu_count')
    elif name == 'network_mode':
        add('IMDS planes', cfg['network']['expected_planes'][ctx.network_mode], '==',
            f'network.expected_planes.{ctx.network_mode}')
    elif name == 'gpu_burn':
        add('per-GPU throughput deviation (%)', cfg[name]['perf_deviation_threshold_pct'], '<=',
            'gpu_burn.perf_deviation_threshold_pct')
    elif name == 'nvbandwidth':
        for metric, value in cfg[name]['tests'].get(ctx.sku, {}).get('thresholds', {}).items():
            add(metric + ' SUM (GB/s)', value, '>=', f'nvbandwidth.tests.{ctx.sku}.thresholds.{metric}')
    elif name == 'babelstream':
        for metric, value in cfg[name]['thresholds'][ctx.sku].items():
            add(metric, value, '>=', f'babelstream.thresholds.{ctx.sku}.{metric}')
    elif name.startswith('rdma_') and name in ('rdma_perf', 'rdma_bw', 'rdma_lat'):
        mode = cfg['rdma_perf'][ctx.network_mode]
        for metric, rule in [('bw_min_gbps', '>='), ('lat_max_us', '<=')]:
            if (name == 'rdma_bw' and metric.startswith('lat')) or (name == 'rdma_lat' and metric.startswith('bw')):
                continue
            add(metric, mode[metric], rule, f'rdma_perf.{ctx.network_mode}.{metric}',
                mode.get('threshold_source', 'config default'))
    elif name.startswith('nccl_'):
        value = result.metrics.get('busbw_min_gbs')
        source = 'PDF example' if 'example reference' in result.metrics.get('threshold_source', '') else 'config default'
        add('largest-size busbw (GB/s), both placements', value, '>=',
            f'nccl.{ctx.network_mode}', source)
    elif name == 'placement':
        add('single-rack maximum hosts', cfg['placement']['rack_max_nodes'], '<=',
            'placement.rack_max_nodes')
    if result.status in ('FAIL', 'WARN'):
        hints = cfg['troubleshooting']
        result.hints = list(dict.fromkeys(result.hints + hints.get(name, hints['default'])))


def performance_values(name, metrics):
    """Explicit performance metrics; avoid baselining temperatures or error counts."""
    if 'hosts' in metrics:
        return {f'host{i}/{key}': value for i, host in enumerate(metrics['hosts'].values())
                for key, value in performance_values(name, host).items()}
    if name == 'gpu_burn':
        return {f'gpu{g["gpu_id"]}/gflops': g['gflops'] for g in metrics.get('per_gpu', [])}
    if name == 'nvbandwidth':
        return {k: v for k, v in metrics.items() if k.endswith('_sum_gbs')}
    if name == 'babelstream':
        return {f'gpu{gpu}/{bench}_gbs': values[0]
                for gpu, benches in metrics.get('per_gpu', {}).items()
                for bench, values in benches.items() if len(values) == 1}
    if name in ('rdma_perf', 'rdma_bw', 'rdma_lat'):
        return {f'{r["server_device"]}/{key}': r[key] for r in metrics.get('rails', [])
                for key in ('bw_gbps', 'lat_us') if r.get(key) is not None}
    if name.startswith('nccl_') and metrics.get('largest'):
        return {label: metrics['largest'][label]['busbw'] for label in ('in_place', 'out_of_place')}
    return {}


def _signature(ctx, result):
    """Reject different workloads even when the check names/scale are identical."""
    section = 'rdma_perf' if result.name.startswith('rdma_') else (
        'nccl' if result.name.startswith('nccl_') else result.name)
    options = dict(ctx.config.get(section, {}))
    # Include all workload options and limits, excluding build locations and reference data.
    for key in ('repository', 'source_dir', 'build_if_missing', 'build_timeout_s',
                'baseline_threshold_pct', 'baseline_tolerance_pct'):
        options.pop(key, None)
    if result.name == 'gpu_burn':
        options['effective_duration_s'] = ctx.burn_duration_s()
    scale = len(result.metrics.get('nodes', ctx.nodes)) or 1
    identity = {'sku': ctx.sku, 'mode': ctx.network_mode, 'scale': scale,
                'gpu_count': ctx.sku_config()['gpu_count'], 'profile': ctx.config.get('_profile'),
                'options': options, 'mock': ctx.mock}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(), scale


def apply_baseline(results, ctx):
    """Save atomically only after the run is known-good; compare all selected checks."""
    from gputest.core import Result

    if not ctx.baseline or ctx.dry_run:
        return
    path = Path(ctx.baseline_file)
    try:
        data = json.loads(path.read_text()) if path.exists() else {'version': 1, 'records': {}}
        if data.get('version') != 1 or not isinstance(data.get('records'), dict):
            raise ValueError('unsupported baseline format; save a new known-good baseline')
        records = data['records']
        observations = {}
        for result in results:
            signature, scale = _signature(ctx, result)
            key = f'{ctx.sku}/{ctx.network_mode}/{result.name}/{scale}'
            values = performance_values(result.name, result.metrics)
            observations[key] = {'signature': signature, 'values': values,
                                 'status': 'PASS' if result.status == 'WARN' and result.metrics.get('uncalibrated')
                                 else result.status}
        if ctx.baseline == 'save':
            bad = [r.name for r in results if r.status == 'FAIL' or
                   (r.status == 'WARN' and not r.metrics.get('uncalibrated')) or
                   (r.status == 'SKIP' and not (r.message.startswith('not applicable') or
                                               r.message.startswith('requires >=2')))]
            if bad or not any(o['values'] for o in observations.values()):
                raise ValueError(f'baseline not saved: run is not known-good ({bad or "no performance data"})')
            if any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0
                   for o in observations.values() for v in o['values'].values()):
                raise ValueError('baseline not saved: invalid performance data')
            records.update(observations)
            data['last_saved_run'] = {'sku': ctx.sku, 'network_mode': ctx.network_mode,
                                      'profile': ctx.config.get('_profile'), 'mock': ctx.mock,
                                      'results_dir': str(ctx.results_dir.resolve())}
            data['saved_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()  # noqa: UP017 (Python 3.10)
            temporary = path.with_suffix(path.suffix + '.tmp')
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
            temporary.replace(path)
            results.append(Result('baseline', 'PASS', f'known-good baseline saved to {path}'))
            return
        for result, (key, observation) in zip(results, observations.items()):
            if result.status == 'SKIP' and result.message.startswith(('not applicable', 'requires >=2')):
                continue
            reference = records.get(key)
            if (not reference or reference.get('signature') != observation['signature']
                    or set(reference.get('values', {})) != set(observation['values'])):
                result.status = 'FAIL'
                result.message += '; baseline missing or incompatible workload'
                continue
            tolerance = ctx.config['baseline']['tolerance_pct_by_check'].get(
                result.name, ctx.config['baseline']['tolerance_pct'])
            if not 0 <= tolerance < 100:
                raise ValueError('baseline tolerance must be in [0, 100)')
            regressions = []
            for metric, value in observation['values'].items():
                old = reference['values'][metric]
                if not isinstance(old, (int, float)) or not math.isfinite(old) or old <= 0:
                    raise ValueError('invalid baseline performance data')
                lower = metric.endswith('/lat_us')
                limit = old * (1 + tolerance / 100 if lower else 1 - tolerance / 100)
                result.thresholds.append({'metric': metric, 'value': limit,
                                          'rule': '<=' if lower else '>=', 'source': 'baseline',
                                          'reference': f'{path}:{key}', 'baseline_value': old})
                if not isinstance(value, (int, float)) or not math.isfinite(value) or (
                        value > limit if lower else value < limit):
                    regressions.append(metric)
            if regressions:
                result.status = 'FAIL'
            elif result.status == 'WARN' and result.metrics.get('uncalibrated'):
                result.status = 'PASS'
                result.metrics['threshold_source'] = 'baseline'
                result.metrics['uncalibrated'] = False
                result.hints = []
                result.message = result.message.replace('threshold unset; calibrate baseline',
                    'using saved baseline').replace('thresholds unset', 'thresholds supplied by baseline')
            elif result.status == 'WARN' and reference.get('status') == 'PASS':
                result.status = 'FAIL'
                result.message += '; status regressed from PASS'
            result.message += (f'; baseline comparison: {regressions or "no performance regression"}'
                               if observation['values'] else '; baseline status checked')
        results.append(Result('baseline', 'FAIL' if any(r.status == 'FAIL' for r in results) else 'PASS',
                              f'compared with {path}'))
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
        results.append(Result('baseline', 'FAIL', str(exc)))


def print_summary(results):
    colors = {'PASS': '32', 'WARN': '33', 'FAIL': '31', 'SKIP': '90'}
    for result in results:
        text = f'[{result.status}] {result.name:<24} {textwrap.shorten(result.message, width=150)}'
        print(f'\033[{colors[result.status]}m{text}\033[0m' if sys.stdout.isatty() else text)
    print(' | '.join(f'{sum(r.status == s for r in results)} {s}' for s in colors))


def write_json(results, run_meta, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'results.json'
    path.write_text(json.dumps({**run_meta, 'checks': [
        {k: v for k, v in vars(r).items() if not k.startswith('_')} for r in results]}, indent=2,
        allow_nan=False) + '\n')
    return path


def _cell(value):
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '&#124;').replace('\n', '<br>')


def write_markdown(results, run_meta, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'results.md'
    lines = [f'# gpuval run — {run_meta.get("run_id", "")}', '']
    lines += [f'- **{key}**: {_cell(value)}' for key, value in run_meta.items()]
    lines += ['', '| Check | Status | Summary | Threshold sources |', '|---|---|---|---|']
    for r in results:
        sources = ', '.join(dict.fromkeys(t['source'] for t in r.thresholds)) or 'No numeric threshold'
        lines.append(f'| {_cell(r.name)} | {r.status} | {_cell(r.message)} | {sources} |')
    for r in results:
        lines += ['', f'## {r.name} — {r.status}', '', r.message, '']
        if r.thresholds:
            lines += ['| Host | Metric | Rule | Value | Source | Reference |', '|---|---|---|---|---|---|']
            for t in r.thresholds:
                lines.append('| ' + _cell(t.get('host', 'all')) + ' | ' + ' | '.join(_cell(t[k]) for k in
                             ('metric', 'rule', 'value', 'source', 'reference')) + ' |')
            lines.append('')
        if r.hints:
            lines += ['Troubleshooting:', ''] + [f'- {hint}' for hint in r.hints] + ['']
        if r.metrics:
            lines += ['| Metric | Value |', '|---|---|']
            lines += [f'| {_cell(k)} | {_cell(v)} |' for k, v in r.metrics.items()]
            lines.append('')
        if r.raw_output_file:
            raw_path = Path(r.raw_output_file).resolve()
            lines += [f'Raw output: [{raw_path.name}]({os.path.relpath(raw_path, out_dir.resolve())})', '']
            if raw_path.exists():
                raw = raw_path.read_text()
                lines += ['````text', raw[:4000].replace('````', '` ` ` `'),
                          '...(truncated; see raw file)' if len(raw) > 4000 else '', '````', '']
    path.write_text('\n'.join(lines) + '\n')
    return path
