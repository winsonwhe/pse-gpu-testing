"""Stress checks: native output, failures, build plans and monitoring without GPUs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from threading import Event

import pytest

from gputest import checks, parsers
from gputest.core import (
    RunContext,
    RunResult,
    load_config,
    run,
    run_checks,
    run_sampled,
)

FIXTURES = Path(__file__).resolve().parent.parent / 'fixtures'


def fixture(name):
    return (FIXTURES / name).read_text()


@pytest.fixture
def ctx(tmp_path):
    config = load_config(FIXTURES.parent / 'config.yaml')
    for key in ('gpu_burn', 'nvbandwidth', 'babelstream'):
        config[key]['build_if_missing'] = False
    return RunContext(config, 'gb300', [], mock=True, results_dir=tmp_path)


def success(stdout=''):
    return RunResult(stdout, '', 0, 0)


def test_native_burn():
    result = parsers.parse_gpu_burn(fixture('gpu_burn_progress.txt'))
    assert len(result['per_gpu']) == 4
    assert result['per_gpu'][0] == {'gpu_id': 0, 'gflops': 9510.0, 'errors': 0, 'temp_c': 61}
    assert not result['errors']


def test_burn_cumulative_errors_and_dead_gpu():
    text = fixture('gpu_burn_progress.txt').replace('errors: 0 - 0', 'errors: 2 - 0')
    parsed = parsers.parse_gpu_burn(text)
    assert parsed['error_count'] == 2  # not 4: counters are cumulative
    assert parsed['errors']
    assert parsers.parse_gpu_burn('GPU 3: FAULTY')['errors']
    assert parsers.parse_gpu_burn('errors: 0 (DIED!)')['errors']


def test_fryer_native():
    parsed = parsers.parse_gpu_burn(fixture('gpu_fryer.txt'))
    assert len(parsed['per_gpu']) == 4
    assert parsed['per_gpu'][0]['gflops'] == 51494
    assert parsed['per_gpu'][0]['temp_c'] == 50


def test_babelstream_native_units():
    parsed = parsers.parse_babelstream(fixture('babelstream_native.txt'))
    assert parsed['benchmarks']['copy'] == pytest.approx([950.12])
    assert len(parsed['benchmarks']) == 5


def test_named_nvbandwidth_sum():
    text = fixture('nvbandwidth_native.txt')
    assert parsers.parse_nvbandwidth(text, 'device_to_device_memcpy_read_sm')['sum_gbs'] == 123.45
    assert parsers.parse_nvbandwidth(text, 'missing')['sum_gbs'] is None
    assert parsers.parse_nvbandwidth('Running first.\nRunning second.\nSUM 12 GB/s',
                                     'first')['sum_gbs'] is None


def test_native_dcgm():
    result = parsers.parse_dcgmi_json(fixture('dcgmi_native.json'))
    assert result['status'] == 'FAIL'
    assert [t['result'] for t in result['tests']] == ['PASS', 'WARN', 'FAIL']
    assert result['tests'][1]['results'][0]['warnings']


@pytest.mark.parametrize('text', ['', 'not json', '{}', '[]', 'null',
                                  '{"DCGM Diagnostic":{"runtime_error":"failed"}}'])
def test_invalid_dcgm_cannot_pass(text):
    assert parsers.parse_dcgmi_json(text)['status'] == 'FAIL'


@pytest.mark.parametrize(('value', 'status'), [('Pass', 'PASS'), ('Warn', 'WARN'),
                                               ('Skip', 'SKIP'), ('unknown', 'WARN')])
def test_dcgm_summary_status(value, status):
    text = json.loads(fixture('dcgmi_native.json'))
    tests = text['DCGM Diagnostic']['test_categories'][0]['tests']
    tests[:] = [{'name': 'test', 'results': [{'status': value}]}]
    assert parsers.parse_dcgmi_json(json.dumps(text))['status'] == status


def test_samples_keep_peak_and_ignore_na():
    peaks = parsers.parse_gpu_samples(fixture('gpu_samples.txt') + '\n0, N/A, nan\n1, N/A, inf')
    assert peaks['0'] == {'temp_c': 65, 'power_w': 1150.75}
    assert peaks['1']['power_w'] == 1140


@pytest.mark.parametrize('output', [
    'garbage',
    fixture('gpu_burn_progress.txt').replace('errors: 0 - 0', 'errors: 1 - 0'),
    fixture('gpu_burn_progress.txt').replace('9510 Gflop/s', '100 Gflop/s'),
    'GPU 0: 0 GFlop/s 0 errors (temp 50)',
])
def test_burn_rejects_bad_results(ctx, monkeypatch, output):
    monkeypatch.setattr('gputest.core.run_sampled', lambda *a, **kw: (success(output), '', ''))
    assert checks.gpu_burn(ctx).status == 'FAIL'


def test_burn_peaks_and_raw_saved(ctx):
    result = run_checks(['gpu_burn'], ctx)[0]
    assert result.status == 'PASS'
    assert result.metrics['peak_temp_c'] == 65
    assert result.metrics['peak_power_w'] == 1150.75
    assert Path(result.raw_output_file).read_text() == fixture('gpu_burn.txt')


def test_fryer_backend(ctx):
    ctx.config['gpu_burn']['backend'] = 'gpu-fryer'
    result = checks.gpu_burn(ctx)
    assert result.status == 'PASS'
    assert not result.metrics['numerical_error_check']


def test_burn_selection_options(ctx, monkeypatch):
    cfg = ctx.config['gpu_burn']
    cfg.update(gpus=[2], double=True, tensor_core=True, memory_percentage=75)
    seen = []

    def sampled(context, cmd, **kwargs):
        seen.append((cmd, kwargs))
        return success('GPU 0: 100 GFlop/s 0 errors (temp 60)'), '2, 60, 500', ''

    monkeypatch.setattr('gputest.core.run_sampled', sampled)
    result = checks.gpu_burn(ctx)
    assert result.status == 'PASS'
    assert result.metrics['per_gpu'][0]['gpu_id'] == 2
    assert '-tc' in seen[0][0] and '-d' in seen[0][0]
    assert '75%' in seen[0][0]
    assert seen[0][1]['env'] == {'CUDA_VISIBLE_DEVICES': '2'}


@pytest.mark.parametrize('name', ['gpu_burn', 'nvbandwidth', 'babelstream', 'dcgmi'])
def test_missing_tools_skip(ctx, monkeypatch, name):
    monkeypatch.setattr(ctx, 'run', lambda *a, **kw: RunResult('', 'command not found', 127, 0))
    assert run_checks([name], ctx)[0].status == 'SKIP'


def test_babelstream_each_gpu_and_threshold(ctx, monkeypatch):
    seen = []

    def execute(cmd, **kwargs):
        seen.append(cmd)
        text = fixture('babelstream_native.txt')
        if cmd[-1] == '3':
            text = text.replace('950120', '850120')
        return success(text)

    monkeypatch.setattr(ctx, 'run', execute)
    result = checks.babelstream(ctx)
    assert result.status == 'FAIL'  # one slow GPU must not be hidden by a mean
    assert [cmd[-1] for cmd in seen] == ['0', '1', '2', '3']
    assert result.metrics['copy_spread_gbs'] == pytest.approx(100)
    assert result.metrics['gpu_count'] == 4


def test_babelstream_incomplete_output_fails(ctx, monkeypatch):
    monkeypatch.setattr(ctx, 'run', lambda *a, **kw: success(''))
    assert checks.babelstream(ctx).status == 'FAIL'


def test_nvbandwidth_threshold_and_missing_sum(ctx, monkeypatch):
    ctx.sku = 'a10'
    monkeypatch.setattr(ctx, 'run', lambda *a, **kw: success(fixture('nvbandwidth_a10.txt')))
    assert checks.nvbandwidth(ctx).status == 'PASS'
    cfg = ctx.config['nvbandwidth']['tests']['a10']
    cfg['thresholds'][cfg['test_names'][0]] = 100
    assert checks.nvbandwidth(ctx).status == 'FAIL'
    monkeypatch.setattr(ctx, 'run', lambda *a, **kw: success(''))
    assert checks.nvbandwidth(ctx).status == 'FAIL'


@pytest.mark.parametrize(('profile', 'level'), [('quick', 1), ('standard', 2), ('full', 3)])
def test_dcgm_profile(ctx, monkeypatch, profile, level):
    ctx.config['_profile'] = profile
    seen = []

    def execute(cmd, **kwargs):
        seen.append((cmd, kwargs))
        return success(fixture('dcgmi_native.json'))

    monkeypatch.setattr(ctx, 'run', execute)
    result = checks.dcgmi(ctx)
    assert seen[0][0][-3:] == ['-r', str(level), '-j']
    assert seen[0][1]['timeout'] == 3600
    assert result.status == 'FAIL'
    assert result.metrics['test_count'] == 3
    assert result.metrics['warn'] == 1


def test_dcgm_failure_exit_retains_tests(ctx, monkeypatch):
    monkeypatch.setattr(ctx, 'run', lambda *a, **kw: RunResult(fixture('dcgmi_native.json'), '', 1, 0))
    result = checks.dcgmi(ctx)
    assert result.status == 'FAIL' and result.metrics['test_count'] == 3


def test_build_dry_run(ctx, capsys):
    ctx.dry_run, ctx.mock = True, False
    for section in ('gpu_burn', 'babelstream', 'nvbandwidth'):
        ctx.config[section]['build_if_missing'] = True
    ctx.config['cuda']['path'] = '/test/cuda'
    results = run_checks(['gpu_burn', 'babelstream', 'nvbandwidth'], ctx)
    assert all(r.status == 'SKIP' for r in results)
    output = capsys.readouterr().out
    assert 'COMPUTE=103' in output and 'CUDAPATH=/test/cuda' in output
    assert '-DCUDA_ARCH=sm_103' in output
    assert '-DCMAKE_CUDA_COMPILER=/test/cuda/bin/nvcc' in output
    assert '-DCMAKE_CUDA_ARCHITECTURES=103' in output
    assert 'git clone' in output


def test_multinode_dry_run(ctx, capsys):
    ctx.dry_run, ctx.nodes = True, ['host1', 'host2']
    cfg = ctx.config['nvbandwidth']
    cfg['multinode'] = True
    cfg['tests']['gb300']['test_names'] = ['multinode_device_to_device_memcpy_read_sm']
    assert checks.nvbandwidth(ctx).status == 'SKIP'
    output = capsys.readouterr().out
    assert '--host host1,host2 --map-by ppr:4:node -np 8' in output


def test_sampling_overlaps_workload(ctx, monkeypatch):
    ctx.mock = False
    started, sampled = Event(), Event()

    def execute(cmd, **kwargs):
        if '--query-gpu=index,temperature.gpu,power.draw' in cmd:
            assert started.wait(2)
            sampled.set()
            return success('0, 60, 500')
        started.set()
        assert sampled.wait(2)
        return success('done')

    monkeypatch.setattr(ctx, 'run', execute)
    rr, samples, error = run_sampled(ctx, ['workload'], timeout=5, fixture='unused')
    assert rr.stdout == 'done' and '0, 60, 500' in samples and not error


def test_sampling_unavailable_warns(ctx, monkeypatch):
    def execute(cmd, **kwargs):
        if cmd[0] == ctx.config['tools']['nvidia_smi']:
            return RunResult('', 'not found', 127, 0)
        return success(fixture('gpu_burn_progress.txt'))

    monkeypatch.setattr(ctx, 'run', execute)
    result = checks.gpu_burn(ctx)
    assert result.status == 'WARN'
    assert 'sampling_warning' in result.metrics
    assert 'peak_temp_c' not in result.metrics


def test_timeout_retains_partial_output(tmp_path, monkeypatch):
    def execute(*a, **kw):
        raise subprocess.TimeoutExpired(['test'], 1, output=b'partial', stderr=b'problem')

    monkeypatch.setattr(subprocess, 'run', execute)
    result = run([sys.executable], timeout=1, results_dir=tmp_path)
    assert result.returncode == -1 and result.stdout == 'partial'
    raw = next(tmp_path.iterdir()).read_text()
    assert 'partial' in raw and 'problem' in raw and 'timeout' in raw


def test_cli_profile_reaches_check(ctx, monkeypatch, tmp_path):
    from gputest import cli

    args = cli.build_parser().parse_args([
        'run', '--sku', 'gb300', '--profile', 'quick', '--tests', 'dcgmi', '--mock',
        '--network-mode', 'single_plane',
    ])
    ctx.config['network']['mode'] = 'single_plane'
    monkeypatch.chdir(tmp_path)
    cli.cmd_run(args, ctx.config)
    report = next(tmp_path.glob('results/*/results.json'))
    assert json.loads(report.read_text())['checks'][0]['metrics']['level'] == 1
    assert json.loads(report.read_text())['checks'][0]['status'] == 'PASS'


def test_build_missing_dependency_skips(ctx, monkeypatch):
    ctx.mock = False
    ctx.config['gpu_burn']['build_if_missing'] = True
    monkeypatch.setattr('shutil.which', lambda cmd: None)
    result = checks.gpu_burn(ctx)
    assert result.status == 'SKIP'
    assert 'nvcc' in result.message


def test_build_failure_is_reported(ctx, monkeypatch):
    ctx.config['gpu_burn']['build_if_missing'] = True
    monkeypatch.setattr(ctx, 'run', lambda *a, **kw: RunResult('', 'build failed', 1, 0))
    result = checks.gpu_burn(ctx)
    assert result.status == 'FAIL' and 'build failed' in result.message


def test_full_dcgm_level_four_override(ctx, monkeypatch):
    ctx.config['_profile'] = 'full'
    ctx.config['dcgmi']['profile_levels']['full'] = 4
    result = checks.dcgmi(ctx)
    assert result.status == 'PASS' and result.metrics['level'] == 4
