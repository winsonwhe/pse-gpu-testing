"""End-to-end profile, report, baseline and preflight failure behavior without GPUs."""
import json
import re
import subprocess
from pathlib import Path
from threading import Barrier

import pytest

from gputest import multihost_checks
from gputest.cli import build_parser, cmd_report, cmd_run
from gputest.core import (
    Result,
    RunContext,
    RunResult,
    load_config,
    run_checks,
    write_json,
    write_markdown,
)

ROOT = Path(__file__).resolve().parents[1]


def context(tmp_path, **kwargs):
    config = load_config(ROOT / 'config.yaml')
    config['network']['mode'] = 'multiplanar'
    return RunContext(config, 'gb300', [], mock=True, network_mode='multiplanar',
                      results_dir=tmp_path, **kwargs)


@pytest.mark.parametrize(('profile', 'duration'), [('quick', 120), ('standard', 1200), ('full', 3600)])
def test_profile_effective_commands_and_membership(tmp_path, capsys, profile, duration):
    ctx = context(tmp_path, dry_run=True)
    ctx.config['_profile'] = profile
    run_checks(['gpu_burn'], ctx)
    out = capsys.readouterr().out
    assert f' {duration}' in out
    assert 'gpu_burn' in ctx.config['profiles'][profile]
    if profile == 'standard':
        assert {'babelstream', 'nvbandwidth', 'dcgmi', 'rdma_perf'} <= set(ctx.config['profiles'][profile])
        assert not any(n.startswith('nccl') for n in ctx.config['profiles'][profile])
    if profile == 'full':
        assert {'nccl_allreduce', 'nccl_alltoall', 'imex'} <= set(ctx.config['profiles'][profile])


def test_all_profiles_have_registered_checks(tmp_path):
    from gputest.core import get_registry
    ctx = context(tmp_path)
    registered = {entry['name'] for entry in get_registry()}
    assert all(set(names) <= registered for names in ctx.config['profiles'].values())


def test_report_sources_hints_escape_and_raw_lines(tmp_path):
    ctx = context(tmp_path)
    results = run_checks(['rdma_perf'], RunContext(ctx.config, 'gb300', ['n1', 'n2'], mock=True,
                                                 network_mode='multiplanar', results_dir=tmp_path))
    assert {t['source'] for t in results[0].thresholds} == {'PDF example'}
    failure = Result('gpu_topology', 'FAIL', 'missing GPU | NIC\nnext', _raw='line one\nline two')
    from gputest.reporting import annotate
    annotate(failure, ctx)
    raw = tmp_path / 'raw.txt'
    raw.write_text(failure._raw)
    failure.raw_output_file = str(raw)
    results.append(failure)
    jpath = write_json(results, {'sku': 'gb300'}, tmp_path)
    mpath = write_markdown(results, {'sku': 'gb300'}, tmp_path)
    data = json.loads(jpath.read_text())
    assert data['checks'][1]['hints'] and data['checks'][1]['thresholds']
    text = mpath.read_text()
    assert 'PDF example' in text and 'config default' in text
    assert 'missing GPU &#124; NIC<br>next' in text
    assert 'line one\nline two' in text  # do not collapse diagnostic lines
    assert 'Troubleshooting:' in text
    assert '[raw.txt](raw.txt)' in text


@pytest.mark.parametrize('name', ['gpu_burn', 'nvbandwidth', 'babelstream'])
def test_baseline_local_performance_regression(tmp_path, monkeypatch, name):
    ctx = context(tmp_path, baseline='save', baseline_file=tmp_path / 'healthy.json')
    results = run_checks([name], ctx)
    assert results[-1].status == 'PASS', results[-1].message
    original = ctx.run
    ctx.baseline = 'compare'
    assert all(r.status == 'PASS' for r in run_checks([name], ctx))

    def degraded(cmd, **kwargs):
        rr = original(cmd, **kwargs)
        if name == 'gpu_burn' and 'gpu_burn' in cmd[0]:
            rr.stdout = re.sub(r'([\d.]+) GFlop/s', lambda m: f'{float(m[1]) * .95:.2f} GFlop/s', rr.stdout)
        elif name == 'nvbandwidth' and 'nvbandwidth' in cmd[0]:
            rr.stdout = rr.stdout.replace('9500.00', '9100.00')
        elif name == 'babelstream' and 'cuda-stream' in cmd[0]:
            rr.stdout = rr.stdout.replace('950', '901')
        return rr

    monkeypatch.setattr(ctx, 'run', degraded)
    # Tighten only comparison tolerance, not the fingerprinted workload.
    ctx.config['baseline']['tolerance_pct_by_check'][name] = 1
    result = run_checks([name], ctx)[0]
    assert result.status == 'FAIL', result.message
    assert any(t['source'] == 'baseline' for t in result.thresholds)
    assert result.hints


def test_failed_save_preserves_previous_snapshot(tmp_path, monkeypatch):
    ctx = context(tmp_path, baseline='save', baseline_file=tmp_path / 'healthy.json')
    run_checks(['gpu_burn'], ctx)
    before = ctx.baseline_file.read_bytes()
    monkeypatch.setattr(ctx, 'run', lambda *a, **k: RunResult('', 'broken tool', 1, 0))
    results = run_checks(['gpu_burn'], ctx)
    assert results[-1].name == 'baseline' and results[-1].status == 'FAIL'
    assert ctx.baseline_file.read_bytes() == before


def test_changed_duration_rejects_baseline(tmp_path):
    ctx = context(tmp_path, baseline='save', baseline_file=tmp_path / 'healthy.json')
    run_checks(['gpu_burn'], ctx)
    ctx.baseline = 'compare'
    ctx.config['profile_settings']['standard']['burn_duration_s'] += 1
    assert 'incompatible workload' in run_checks(['gpu_burn'], ctx)[0].message


def test_remote_per_host_workloads_overlap(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    ctx.nodes = ['n1', 'n2']
    barrier = Barrier(2)
    seen = []
    # Registry retains its function reference, so exercise the command boundary.
    from gputest.core import run

    def execute(cmd, **kwargs):
        if cmd[0].endswith('gpu_burn'):
            barrier.wait(timeout=3)
            seen.append(kwargs['node'])
        return run(cmd, mock=True, mock_fixture=kwargs.get('mock_fixture'),
                   results_dir=kwargs.get('results_dir'))

    monkeypatch.setattr('gputest.core.run', execute)
    result = run_checks(['gpu_burn'], ctx)[0]
    assert result.status == 'PASS', result.message
    assert set(seen) == {'n1', 'n2'}
    assert set(result.metrics['hosts']) == {'n1', 'n2'}


@pytest.mark.parametrize('problem', ['missing_vrf', 'unbound_vf'])
def test_vrf_failure_produces_targeted_hint(tmp_path, monkeypatch, problem):
    ctx = context(tmp_path)
    original = ctx.run

    def execute(cmd, **kwargs):
        rr = original(cmd, **kwargs)
        if cmd[:4] == ['ip', '-j', '-d', 'link']:
            links = json.loads(rr.stdout)
            if problem == 'missing_vrf':
                links = [link for link in links if link['ifname'] != 'vrf_r0']
            else:
                links[1]['master'] = 'wrong'
            rr.stdout = json.dumps(links)
        return rr

    monkeypatch.setattr(ctx, 'run', execute)
    result = run_checks(['rdma_iface'], ctx)[0]
    assert result.status == 'FAIL'
    assert 'VRF' in result.message or 'unbound VF' in result.message
    assert any('enslaved' in hint for hint in result.hints)


def test_wrong_planes_block_burn_and_produce_report(tmp_path):
    ctx = context(tmp_path)
    ctx.network_mode = 'single_plane'  # mock IMDS reports 4
    results = run_checks(['network_mode', 'gpu_burn'], ctx)
    assert results[0].status == 'FAIL' and 'wrong plane count' in results[0].message
    assert results[1].status == 'SKIP' and 'blocked' in results[1].message


def test_cli_reports_failure_with_nonzero_status_and_rerenders(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    args = build_parser().parse_args(['run', '--sku', 'gb300', '--mock', '--tests', 'gpu_topology',
                                     '--output-dir', str(tmp_path)])
    monkeypatch.setattr('gputest.parsers.parse_gpu_topology', lambda text: {'gpus': [], 'nics': []})
    assert cmd_run(args, ctx.config) == 1
    data = json.loads((tmp_path / 'results.json').read_text())
    assert data['checks'][0]['hints']
    (tmp_path / 'results.md').unlink()
    cmd_report(type('Args', (), {'result_dir': str(tmp_path)})(), ctx.config)
    assert 'Troubleshooting:' in (tmp_path / 'results.md').read_text()


def test_full_dry_run_cannot_execute_or_write_baseline(tmp_path, monkeypatch):
    ctx = context(tmp_path, baseline='save', baseline_file=tmp_path / 'baseline.json', dry_run=True)
    ctx.nodes = ['n1', 'n2']
    ctx.config['_profile'] = 'full'

    def forbidden(*args, **kwargs):
        pytest.fail('dry-run executed a subprocess')

    monkeypatch.setattr(subprocess, 'run', forbidden)
    assert all(r.status == 'SKIP' for r in run_checks(ctx.config['profiles']['full'], ctx))
    assert not ctx.baseline_file.exists()
    assert not list(tmp_path.iterdir())


def test_calibration_baseline_does_not_hide_later_operational_warning(tmp_path, monkeypatch):
    ctx = context(tmp_path, baseline='save', baseline_file=tmp_path / 'healthy.json')
    ctx.nodes = ['n1', 'n2']
    run_checks(['nccl_allreduce'], ctx)  # unconfigured threshold is the sole initial warning
    ctx.baseline = 'compare'
    result = run_checks(['nccl_allreduce'], ctx)[0]
    assert result.status == 'PASS' and not result.hints
    assert 'using saved baseline' in result.message
    original = multihost_checks._nccl
    from gputest.core import get_registry
    entry = next(e for e in get_registry() if e['name'] == 'nccl_allreduce')

    def warning(context):
        result = original(context, 'allreduce')
        result.status = 'WARN'
        result.metrics['uncalibrated'] = False
        result.message = 'operational placement warning'
        return result

    monkeypatch.setitem(entry, 'fn', warning)
    result = run_checks(['nccl_allreduce'], ctx)[0]
    assert result.status == 'FAIL' and 'status regressed' in result.message


def test_unknown_check_fails_and_missing_tool_cannot_be_saved(tmp_path, monkeypatch):
    ctx = context(tmp_path, baseline='save', baseline_file=tmp_path / 'healthy.json')
    assert run_checks(['typo'], ctx)[0].status == 'FAIL'
    monkeypatch.setattr(ctx, 'run', lambda *a, **k: RunResult('', 'missing', 127, 0))
    result = run_checks(['gpu_burn'], ctx)[-1]
    assert result.status == 'FAIL' and not ctx.baseline_file.exists()
