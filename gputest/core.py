"""Core: Result type, check registry, run() helper, report writer."""

from __future__ import annotations

import datetime
import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event
from typing import Any

import yaml

from gputest.reporting import (
    annotate,
    apply_baseline,
    print_summary,
    write_json,
    write_markdown,
)

__all__ = ['print_summary', 'write_json', 'write_markdown']

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

VALID_STATUSES = {"PASS", "WARN", "FAIL", "SKIP"}


@dataclass
class Result:
    name: str
    status: str          # PASS | WARN | FAIL | SKIP
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    raw_output_file: str | None = None
    # internal: raw text kept in memory until written
    _raw: str = field(default="", repr=False)
    thresholds: list[dict] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {self.status!r}")

    def passed(self) -> bool:
        return self.status == "PASS"


# ---------------------------------------------------------------------------
# Check registry
# ---------------------------------------------------------------------------

_REGISTRY: list[dict] = []


def check(
    name: str,
    *,
    category: str,
    skus: list[str] | None = None,
    requires_multi_host: bool = False,
):
    """Decorator that registers a check function."""
    def decorator(fn):
        _REGISTRY.append({
            "name": name,
            "category": category,
            "skus": skus,          # None = all SKUs
            "multi_host": requires_multi_host,
            "fn": fn,
        })
        return fn
    return decorator


def get_registry() -> list[dict]:
    return list(_REGISTRY)


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    stdout: str
    stderr: str
    returncode: int
    elapsed_s: float
    skipped: bool = False   # set when dry_run=True


# ---------------------------------------------------------------------------
# RunContext
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    config: dict
    sku: str
    nodes: list[str]          # empty list = localhost only
    dry_run: bool = False
    mock: bool = False
    network_mode: str = "single_plane"
    baseline: str | None = None
    baseline_file: Path = field(default_factory=lambda: Path("baseline.json"))
    node_count: int | None = None
    target_node: str | None = None
    results_dir: Path = field(default_factory=lambda: Path("results/current"))

    def run(
        self,
        cmd: list[str],
        *,
        node: str | None = None,
        sudo: bool = False,
        timeout: int = 120,
        mock_fixture: str | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> RunResult:
        return run(
            cmd,
            node=node or self.target_node,
            sudo=sudo,
            timeout=timeout,
            dry_run=self.dry_run,
            mock=self.mock,
            mock_fixture=mock_fixture,
            results_dir=self.results_dir,
            cwd=cwd,
            env=env,
            ssh_tool=self.config.get("tools", {}).get("ssh", "ssh"),
            remote_timeout_tool=self.config.get("tools", {}).get("timeout", "timeout"),
        )

    def run_parallel(self, jobs: list[dict]) -> list[RunResult]:
        """Bounded parallel local/SSH calls; dry-run output stays in input order."""
        if self.dry_run:
            return [self.run(**job) for job in jobs]
        workers = self.config.get("executor", {}).get("parallelism", 16)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda job: self.run(**job), jobs))

    def run_pair(self, server: dict, client: dict, port: int) -> tuple[RunResult, RunResult]:
        """Start a bounded server; inspect its listening socket without consuming it."""
        cfg = self.config["rdma_perf"]
        ready = {"cmd": [self.config["tools"]["ss"], "-H", "-ltn", "sport", "=", f":{port}"],
                     "node": server["node"], "timeout": cfg["probe_timeout_s"],
                     "mock_fixture": "rdma_listen.txt"}
        if self.dry_run:
            sr = self.run(**server)
            self.run(**ready)
            return sr, self.run(**client)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.run, **server)
            for _ in range(cfg["ready_attempts"]):
                probe = self.run(**ready)
                if probe.returncode == 0 and probe.stdout.strip():
                    break
                if future.done():
                    return future.result(), RunResult("", "server exited before listening", 1, 0)
                Event().wait(cfg["ready_interval_s"])
            else:
                return future.result(), RunResult("", "server readiness timed out", 1, 0)
            client_result = self.run(**client)
            return future.result(), client_result

    def burn_duration_s(self):
        profile = self.config.get('_profile', 'standard').removeprefix('a10_')
        return self.config['profile_settings'][profile]['burn_duration_s']

    def sku_config(self) -> dict:
        return self.config.get("skus", {}).get(self.sku, {})

    def network_config(self) -> dict:
        mode = self.config.get("network", {}).get(self.network_mode, {})
        return {**mode, **mode.get(self.sku, {})}


# ---------------------------------------------------------------------------
# run() helper
# ---------------------------------------------------------------------------

def run(
    cmd: list[str],
    *,
    node: str | None = None,
    sudo: bool = False,
    timeout: int = 120,
    dry_run: bool = False,
    mock: bool = False,
    mock_fixture: str | None = None,
    results_dir: Path | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    remote_timeout_tool: str = "timeout",
    ssh_tool: str = "ssh",
) -> RunResult:
    """
    Execute a command, save its output, return a RunResult.

    - dry_run: print the command, return empty result.
    - mock: read fixtures/<mock_fixture> instead of running.
    - node: SSH to that host.
    - sudo: prepend sudo.
    """
    fixture_command = cmd[0]
    if sudo:
        cmd = ["sudo", "-n"] + cmd
    if node:
        if node.startswith("-") or any(c.isspace() for c in node):
            raise ValueError("invalid SSH host")
        remote = [remote_timeout_tool, "--signal=TERM", "--kill-after=5", str(timeout)]
        if env:
            remote += ["env"] + [f"{k}={v}" for k, v in env.items()]
        remote += cmd
        script = shlex.join(remote)
        if cwd:
            script = f"cd {shlex.quote(str(cwd))} && {script}"
        cmd = [ssh_tool, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--", node, script]
        cwd, env = None, None

    if dry_run:
        prefix = f"cd {shlex.quote(str(cwd))} && " if cwd else ""
        assignments = ' '.join(f'{k}={shlex.quote(v)}' for k, v in (env or {}).items())
        print(f"[dry-run] {prefix}{assignments + ' ' if assignments else ''}{shlex.join(cmd)}")
        return RunResult(stdout="", stderr="", returncode=0, elapsed_s=0.0, skipped=True)

    if mock:
        fixture = mock_fixture or fixture_command
        fixture_root = Path(__file__).resolve().parent.parent / "fixtures"
        fixture_path = fixture_root / fixture
        if not fixture_path.exists():
            fixture_path = fixture_root / (fixture + ".txt")
        if fixture_path.exists():
            stdout = fixture_path.read_text()
        else:
            error = f'mock fixture unavailable: {fixture}'
            _save_raw(cmd, '', error, results_dir)
            return RunResult('', error, 127, 0)
        rr = RunResult(stdout=stdout, stderr="", returncode=0, elapsed_s=0.0)
        _save_raw(cmd, stdout, "", results_dir)
        return rr

    import time
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env={**os.environ, **env} if env else None,
        )
        elapsed = time.monotonic() - t0
        _save_raw(cmd, proc.stdout, proc.stderr, results_dir)
        return RunResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            elapsed_s=elapsed,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - t0
        msg = f"[timeout after {timeout}s]"
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        _save_raw(cmd, stdout, stderr + msg, results_dir)
        return RunResult(stdout=stdout, stderr=stderr + msg, returncode=-1, elapsed_s=elapsed)
    except FileNotFoundError:
        _save_raw(cmd, "", f"command not found: {cmd[0]}", results_dir)
        return RunResult(stdout="", stderr=f"command not found: {cmd[0]}",
                         returncode=127, elapsed_s=0.0)


def run_sampled(ctx, cmd, *, timeout, fixture, cwd=None, env=None):
    """Run a workload while issuing bounded nvidia-smi samples through run()."""
    cfg = ctx.config['gpu_burn']
    sample_cmd = [ctx.config['tools']['nvidia_smi'],
                  '--query-gpu=index,temperature.gpu,power.draw',
                  '--format=csv,noheader,nounits']
    samples, errors = [], []
    stop = Event()

    def sample():
        while True:
            rr = ctx.run(sample_cmd, timeout=cfg['sample_timeout_s'],
                         mock_fixture='gpu_samples.txt')
            samples.append(rr.stdout)
            if rr.returncode:
                errors.append('temperature/power sampling unavailable: ' + rr.stderr.strip())
                break
            if ctx.mock or ctx.dry_run or stop.wait(cfg['sample_interval_s']):
                break

    with ThreadPoolExecutor(max_workers=1) as pool:
        sampler = pool.submit(sample)
        try:
            rr = ctx.run(cmd, timeout=timeout, mock_fixture=fixture, cwd=cwd, env=env)
        finally:
            stop.set()
        sampler.result()
    return rr, '\n'.join(samples), '; '.join(errors)


def _save_raw(cmd: list[str], stdout: str, stderr: str, results_dir: Path | None):
    if not results_dir:
        return
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%H%M%S_%f")
    safe = cmd[0].replace("/", "_").lstrip("_")
    path = results_dir / f"{safe}_{ts}.txt"
    with open(path, "w") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n")
        if stdout:
            f.write(stdout)
        if stderr:
            f.write("\n# stderr:\n")
            f.write(stderr)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_checks(
    checks: list[str],
    ctx: RunContext,
) -> list[Result]:
    """Run the requested checks in order, return results."""
    registry = {c["name"]: c for c in _REGISTRY}
    results = []

    for name in checks:
        entry = registry.get(name)
        if not entry:
            results.append(Result(name=name, status="FAIL", message="check not found in registry"))
            continue

        # SKU filter
        if entry["skus"] and ctx.sku not in entry["skus"]:
            results.append(Result(
                name=name, status="SKIP",
                message=f"not applicable to SKU {ctx.sku!r}",
            ))
            continue

        # Multi-host filter
        if entry["multi_host"] and len(ctx.nodes) < 2:
            results.append(Result(
                name=name, status="SKIP",
                message="requires >=2 hosts (pass --nodes host1,host2,...)",
            ))
            continue

        if entry['category'] != 'preflight' and any(
                r.name == 'network_mode' and r.status == 'FAIL' for r in results):
            results.append(Result(name, 'SKIP', 'blocked by failed network_mode preflight'))
            continue
        try:
            # Workloads/preflights run on every requested host. Coordinating checks
            # already consume ctx.nodes and must run only once.
            if ctx.nodes and not entry['multi_host'] and name not in ('network_mode', 'rdma_iface') and not (
                    name == 'nvbandwidth' and ctx.config['nvbandwidth']['multinode']):
                def run_host(item, check_name=name):
                    index, node = item
                    child = replace(ctx, nodes=[], target_node=node, baseline=None,
                                    results_dir=ctx.results_dir / f'host-{index}')
                    return run_checks([check_name], child)[0]

                if ctx.dry_run:
                    host_results = list(map(run_host, enumerate(ctx.nodes)))
                else:
                    with ThreadPoolExecutor(max_workers=ctx.config['executor']['parallelism']) as pool:
                        host_results = list(pool.map(run_host, enumerate(ctx.nodes)))
                status = max((r.status for r in host_results), key=['PASS', 'SKIP', 'WARN', 'FAIL'].index)
                result = Result(name, status, '; '.join(f'{n}: {r.message}' for n, r in zip(ctx.nodes, host_results)),
                                {'nodes': ctx.nodes, 'hosts': {n: r.metrics for n, r in zip(ctx.nodes, host_results)}},
                                _raw='\n'.join(f'{n}: {r.status}\n{r._raw}' for n, r in zip(ctx.nodes, host_results)))
                result.thresholds = [dict(t, host=n) for n, r in zip(ctx.nodes, host_results) for t in r.thresholds]
                result.hints = list(dict.fromkeys(h for r in host_results for h in r.hints))
                if all(r.metrics.get('uncalibrated') for r in host_results):
                    result.metrics['uncalibrated'] = True
            else:
                result = entry["fn"](ctx)
        except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
            result = Result(name=name, status="FAIL", message=f"exception: {exc}")

        if result._raw and not ctx.dry_run:
            ctx.results_dir.mkdir(parents=True, exist_ok=True)
            path = ctx.results_dir / f"{name}.txt"
            path.write_text(result._raw)
            result.raw_output_file = str(path.resolve())
        results.append(result)

    for result in results:
        if not result.thresholds:
            annotate(result, ctx)
    apply_baseline(results, ctx)
    for result in results:
        if result.status in ('WARN', 'FAIL'):
            result.hints = list(dict.fromkeys(result.hints + ctx.config['troubleshooting'].get(
                result.name, ctx.config['troubleshooting']['default'])))
    return results
