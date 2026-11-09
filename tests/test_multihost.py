"""CPU-only validation of multi-host planning, parsers, failures and baselines."""
import json
import shlex
import subprocess
from pathlib import Path
from threading import Barrier

import pytest

import gputest.checks  # noqa: F401
from gputest import parsers
from gputest.cli import _detect_network_mode, build_parser
from gputest.core import RunContext, RunResult, load_config, run, run_checks
from gputest.multihost_checks import select_nodes

ROOT = Path(__file__).resolve().parents[1]


def context(tmp_path, mode='multiplanar', sku='gb300', **kwargs):
    return RunContext(load_config(ROOT / 'config.yaml'), sku, ['n1', 'n2'], mock=True,
                      network_mode=mode, results_dir=tmp_path, **kwargs)


def test_real_nccl_samples():
    for collective, expected in [('allreduce', 841.63), ('alltoall', 627.69)]:
        parsed = parsers.parse_nccl((ROOT / f'fixtures/nccl_{collective}.txt').read_text())
        assert parsed['largest']['size'] == 17179869184
        assert parsed['largest']['out_of_place']['busbw'] == expected
        assert len(parsed['rows']) == 2
    assert parsers.parse_nccl('17179869184 invalid table')['largest'] is None


def test_perftest_average_not_peak_or_median():
    for kind, expected in [('bw', 459), ('lat', 3.9)]:
        text = (ROOT / f'fixtures/ib_write_{kind}.txt').read_text()
        assert parsers.parse_perftest(text, kind)['value'] == expected
        assert parsers.parse_perftest('tool failed', kind)['value'] is None


def test_remote_quoting_environment_cwd_and_timeout(monkeypatch, tmp_path):
    captured = {}

    def execute(cmd, **kwargs):
        captured.update(cmd=cmd, kwargs=kwargs)
        return subprocess.CompletedProcess(cmd, 0, 'ok', '')

    monkeypatch.setattr(subprocess, 'run', execute)
    run(['printf', '%s', 'a b; $(false)'], node='n1', env={'X': 'a b'},
        cwd='/tmp/a b', timeout=12, results_dir=tmp_path)
    argv = captured['cmd']
    assert argv[-2] == 'n1'
    script = argv[-1]
    assert "cd '/tmp/a b' && timeout --signal=TERM --kill-after=5 12" in script
    assert shlex.split(script.split(' && ')[1])[-1] == 'a b; $(false)'
    assert "'X=a b'" in script
    assert captured['kwargs']['env'] is None
    assert captured['kwargs']['cwd'] is None
    assert list(tmp_path.glob('ssh_*.txt'))


def test_timeout_keeps_partial_raw(monkeypatch, tmp_path):
    def execute(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, output=b'partial', stderr=b'error')

    monkeypatch.setattr(subprocess, 'run', execute)
    rr = run(['work'], node='n1', timeout=1, results_dir=tmp_path)
    assert rr.returncode == -1 and rr.stdout == 'partial'
    assert 'timeout' in rr.stderr
    assert 'partial' in next(tmp_path.glob('*.txt')).read_text()


def test_parallel_ssh_not_serial(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    barrier = Barrier(2)

    def execute(cmd, **kwargs):
        barrier.wait(timeout=2)
        return RunResult(cmd[0], '', 0, 0)

    monkeypatch.setattr(ctx, 'run', execute)
    results = ctx.run_parallel([{'cmd': [name], 'node': name} for name in ('n1', 'n2')])
    assert [r.stdout for r in results] == ['n1', 'n2']


def test_multiplanar_rdma_and_spread(tmp_path):
    result = run_checks(['rdma_perf'], context(tmp_path))[0]
    assert result.status == 'PASS', result.message
    assert len(result.metrics['rails']) == 4
    assert result.metrics['bw_gbps_spread_pct'] == 0
    assert result.metrics['rails'][0]['lat_us'] == 3.9
    assert list(tmp_path.glob('ssh_*.txt'))
    assert result.raw_output_file


def test_single_plane_discovers_devices_without_vrf(tmp_path):
    ctx = context(tmp_path, 'single_plane', 'a10')
    result = run_checks(['rdma_perf'], ctx)[0]
    assert result.status == 'WARN', result.message
    assert len(result.metrics['rails']) == 2
    commands = '\n'.join(p.read_text().splitlines()[0] for p in tmp_path.glob('ssh_*.txt'))
    assert 'vrf exec' not in commands
    assert '--ipv6' not in commands
    assert '10.0.0.1' in commands


def test_dry_run_all_rails_and_env_no_execution(tmp_path, capsys, monkeypatch):
    ctx = context(tmp_path, dry_run=True)

    def forbidden(*args, **kwargs):
        pytest.fail('dry-run attempted a subprocess')

    monkeypatch.setattr(subprocess, 'run', forbidden)
    results = run_checks(['rdma_perf', 'nccl_allreduce', 'nccl_alltoall'], ctx)
    assert all(r.status == 'SKIP' for r in results), [r.message for r in results]
    output = capsys.readouterr().out
    assert output.count('/usr/bin/ib_write_bw') == 8
    assert output.count('/usr/bin/ib_write_lat') == 8
    assert 'vrf_r3 numactl -N 1' in output
    assert 'NCCL_TUNING_VARIANT=final_small_cta16_qps16' in output
    assert 'UCX_NET_DEVICES=eth0' in output
    assert '--mca pml ucx' in output
    assert '1G -e 16G' in output
    assert not list(tmp_path.iterdir())


def test_server_failure_cannot_pass(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    monkeypatch.setattr(ctx, 'run_pair', lambda *args:
                        (RunResult('', 'server failed', 1, 0), RunResult('', '', 0, 0)))
    assert run_checks(['rdma_perf'], ctx)[0].status == 'FAIL'


def test_pair_waits_for_readiness_and_returns_failure(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    seen = []

    def execute(cmd, **kwargs):
        seen.append(cmd[0])
        if cmd[0] == 'server':
            return RunResult('', 'failure', 1, 0)
        return RunResult('', '', 0, 0)

    monkeypatch.setattr(ctx, 'run', execute)
    sr, cr = ctx.run_pair({'cmd': ['server'], 'node': 'n1'},
                         {'cmd': ['client'], 'node': 'n2'}, 18515)
    assert sr.returncode == cr.returncode == 1
    assert 'client' not in seen


def test_placement_and_unknown_metadata():
    inv = {'a': {'gmf': 'g1', 'local_block': 'b1'},
           'b': {'gmf': 'g1', 'local_block': 'b1'},
           'c': {'gmf': 'g2', 'local_block': 'b2'}}
    assert select_nodes(inv, 2, 18) == ['a', 'b']
    with pytest.raises(ValueError, match='single GMF'):
        select_nodes(inv, 3, 18)
    with pytest.raises(ValueError, match='metadata'):
        select_nodes({'a': {}, 'b': {}}, 2, 18)
    cross = {f'n{i}': {'gmf': f'g{i//2}', 'local_block': 'b1' if i < 4 else 'b2'}
             for i in range(6)}
    assert select_nodes(cross, 4, 2) == ['n0', 'n1', 'n2', 'n3']


def test_nccl_thresholds_both_columns_and_incomplete_table(tmp_path, monkeypatch):
    ctx = context(tmp_path, 'single_plane', 'a10')
    ctx.config['nccl']['single_plane']['thresholds']['allreduce'] = 842
    result = run_checks(['nccl_allreduce'], ctx)[0]
    assert result.status == 'FAIL'  # out-of-place 841.63, in-place 842.11
    assert result.metrics['env'] == {'NCCL_IB_HCA': 'mlx5_0,mlx5_1'}
    assert result.metrics['ranks'] == 8
    original = ctx.run

    def execute(cmd, **kwargs):
        rr = original(cmd, **kwargs)
        if 'mpirun' in cmd:
            rr.stdout = rr.stdout.replace('17179869184', '4294967296')
        return rr

    monkeypatch.setattr(ctx, 'run', execute)
    result = run_checks(['nccl_allreduce'], ctx)[0]
    assert 'not completed' in result.message


def test_large_cluster_tuning_and_mpirun(tmp_path, capsys):
    ctx = context(tmp_path, dry_run=True)
    ctx.nodes = [f'n{i}' for i in range(33)]
    result = run_checks(['nccl_allreduce'], ctx)[0]
    assert result.status == 'SKIP', result.message
    assert result.metrics['ranks'] == 132
    assert result.metrics['env']['NCCL_MAX_CTAS'] == '32'
    assert result.metrics['env']['NCCL_IB_QPS_PER_CONNECTION'] == '1'
    output = capsys.readouterr().out
    assert 'mpirun -np 132 --host' in output
    assert '--map-by ppr:4:node' in output
    assert 'n0:4,n1:4' in output


def test_imex_rejects_cross_gmf_membership(tmp_path):
    ctx = context(tmp_path)
    assert run_checks(['imex'], ctx)[0].status == 'PASS'
    ctx.config['placement']['inventory'] = {'n1': {'gmf': 'g1'}, 'n2': {'gmf': 'g2'}}
    assert run_checks(['imex'], ctx)[0].status == 'FAIL'
    ctx.sku = 'b300'
    assert run_checks(['imex'], ctx)[0].status == 'SKIP'


def test_baseline_save_compare_and_regression(tmp_path, monkeypatch):
    ctx = context(tmp_path, baseline='save', baseline_file=tmp_path / 'baseline.json')
    assert run_checks(['rdma_perf', 'nccl_allreduce'], ctx)[0].status == 'PASS'
    assert len(json.loads(ctx.baseline_file.read_text())['records']) == 2
    ctx.baseline = 'compare'
    assert all(r.status == 'PASS' for r in run_checks(['rdma_perf', 'nccl_allreduce'], ctx))
    original = ctx.run_pair

    def regress(*args):
        sr, cr = original(*args)
        cr.stdout = cr.stdout.replace('459.00', '350.00').replace('3.90', '5.00')
        return sr, cr

    monkeypatch.setattr(ctx, 'run_pair', regress)
    result = run_checks(['rdma_perf'], ctx)[0]
    assert result.status == 'FAIL'
    assert 'baseline comparison' in result.message


def test_detection_runs_on_target_and_dry_run(tmp_path, capsys):
    ctx = context(tmp_path)
    assert _detect_network_mode(ctx.config, ctx) == 'multiplanar'
    ctx.dry_run = True
    assert _detect_network_mode(ctx.config, ctx) == 'single_plane'
    assert 'fallback_mode' in capsys.readouterr().out
    args = build_parser().parse_args(['run', '--sku', 'gb300', '--baseline', 'save', '--node-count', '2'])
    assert args.baseline == 'save' and args.node_count == 2


def test_example_scale_reference_uses_eighty_percent(tmp_path):
    ctx = context(tmp_path, 'single_plane', 'a10')
    mode = ctx.config['nccl']['single_plane']
    mode['example_reference_gbs']['allreduce'][2] = 1000
    result = run_checks(['nccl_allreduce'], ctx)[0]
    assert result.status == 'PASS', result.message
    assert result.metrics['busbw_min_gbs'] == 800
    assert 'example reference' in result.message


def test_nccl_validation_error_fails_even_small_rows(tmp_path, monkeypatch):
    ctx = context(tmp_path, 'single_plane', 'a10')
    original = ctx.run

    def execute(cmd, **kwargs):
        rr = original(cmd, **kwargs)
        if 'mpirun' in cmd:
            rr.stdout = rr.stdout.replace('838.38      0', '838.38      1')
        return rr

    monkeypatch.setattr(ctx, 'run', execute)
    assert 'validation errors' in run_checks(['nccl_allreduce'], ctx)[0].message


def test_ipv6_single_plane_only_when_detected(tmp_path):
    ctx = context(tmp_path, 'single_plane', 'a10')
    ctx.config['rdma_perf']['single_plane']['ip_family'] = 'inet6'
    result = run_checks(['rdma_bw'], ctx)[0]
    assert result.status == 'WARN', result.message
    commands = '\n'.join(p.read_text().splitlines()[0] for p in tmp_path.glob('ssh_*.txt'))
    assert '--ipv6 --ipv6-addr' in commands
    assert 'vrf exec' not in commands


def test_nccl_stops_before_launch_on_imex_failure(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    original = ctx.run
    launched = []

    def execute(cmd, **kwargs):
        if 'mpirun' in cmd:
            launched.append(cmd)
        if cmd[0] == 'systemctl':
            return RunResult('inactive', '', 3, 0)
        return original(cmd, **kwargs)

    monkeypatch.setattr(ctx, 'run', execute)
    result = run_checks(['nccl_allreduce'], ctx)[0]
    assert result.status == 'FAIL' and 'inactive' in result.message
    assert not launched


def test_missing_baseline_and_invalid_node_count_fail(tmp_path):
    ctx = context(tmp_path, baseline='compare', baseline_file=tmp_path / 'missing.json')
    assert 'baseline missing' in run_checks(['rdma_perf'], ctx)[0].message
    ctx.node_count = 0
    assert run_checks(['nccl_allreduce'], ctx)[0].status == 'FAIL'


def test_ip_parser_excludes_tentative_addresses():
    parsed = parsers.parse_ip_addresses('[{"ifname":"eth0","addr_info":['
                                       '{"family":"inet6","local":"fd00::1","tentative":true},'
                                       '{"family":"inet","local":"10.0.0.1"}]}]')
    assert parsed['eth0'] == [{'family': 'inet', 'local': '10.0.0.1'}]


def test_imex_only_validates_selected_nccl_hosts(tmp_path, monkeypatch):
    ctx = context(tmp_path, node_count=2)
    ctx.nodes += ['n3', 'n4']
    ctx.config['placement']['inventory'] = {
        n: {'gmf': 'g1' if n in ('n1', 'n2') else 'g2'} for n in ctx.nodes}
    original = ctx.run
    service_nodes = []

    def execute(cmd, **kwargs):
        if cmd[0] == 'systemctl':
            service_nodes.append(kwargs['node'])
        return original(cmd, **kwargs)

    monkeypatch.setattr(ctx, 'run', execute)
    result = run_checks(['nccl_allreduce'], ctx)[0]
    assert result.status == 'WARN', result.message
    assert set(service_nodes) == {'n1', 'n2'}
