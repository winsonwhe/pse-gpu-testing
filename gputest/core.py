"""Core: Result type, check registry, run() helper, report writer."""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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
    results_dir: Path = field(default_factory=lambda: Path("results/current"))

    def run(
        self,
        cmd: list[str],
        *,
        node: str | None = None,
        sudo: bool = False,
        timeout: int = 120,
        mock_fixture: str | None = None,
    ) -> RunResult:
        return run(
            cmd,
            node=node,
            sudo=sudo,
            timeout=timeout,
            dry_run=self.dry_run,
            mock=self.mock,
            mock_fixture=mock_fixture,
            results_dir=self.results_dir,
        )

    def sku_config(self) -> dict:
        return self.config.get("skus", {}).get(self.sku, {})

    def network_config(self) -> dict:
        return (
            self.config.get("network", {})
            .get(self.network_mode, {})
            .get(self.sku, {})
        )


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
) -> RunResult:
    """
    Execute a command, save its output, return a RunResult.

    - dry_run: print the command, return empty result.
    - mock: read fixtures/<mock_fixture> instead of running.
    - node: SSH to that host.
    - sudo: prepend sudo.
    """
    if sudo:
        cmd = ["sudo"] + cmd
    if node:
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", node] + cmd

    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return RunResult(stdout="", stderr="", returncode=0, elapsed_s=0.0, skipped=True)

    if mock:
        fixture = mock_fixture or (cmd[0] if not node else cmd[6])
        fixture_path = Path("fixtures") / fixture
        if not fixture_path.exists():
            fixture_path = Path("fixtures") / (fixture + ".txt")
        if fixture_path.exists():
            stdout = fixture_path.read_text()
        else:
            stdout = ""
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
        )
        elapsed = time.monotonic() - t0
        _save_raw(cmd, proc.stdout, proc.stderr, results_dir)
        return RunResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            elapsed_s=elapsed,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        msg = f"[timeout after {timeout}s]"
        _save_raw(cmd, msg, "", results_dir)
        return RunResult(stdout=msg, stderr="", returncode=-1, elapsed_s=elapsed)
    except FileNotFoundError:
        _save_raw(cmd, "", f"command not found: {cmd[0]}", results_dir)
        return RunResult(stdout="", stderr=f"command not found: {cmd[0]}",
                         returncode=127, elapsed_s=0.0)


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
            results.append(Result(name=name, status="SKIP", message="check not found in registry"))
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

        try:
            result = entry["fn"](ctx)
        except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
            result = Result(name=name, status="FAIL", message=f"exception: {exc}")

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

_STATUS_ICON = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}
_STATUS_COLOR = {
    "PASS": "\033[32m",  # green
    "WARN": "\033[33m",  # yellow
    "FAIL": "\033[31m",  # red
    "SKIP": "\033[90m",  # dark grey
}
_RESET = "\033[0m"


def _color(status: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{_STATUS_COLOR.get(status, '')}{text}{_RESET}"


def print_summary(results: list[Result]) -> None:
    counts = {s: 0 for s in VALID_STATUSES}
    for r in results:
        icon = _STATUS_ICON.get(r.status, r.status)
        line = f"{icon} {r.name:<30} {r.message}"
        print(_color(r.status, line))
        counts[r.status] += 1
    print("-" * 60)
    parts = [f"{counts[s]} {s}" for s in ("PASS", "FAIL", "WARN", "SKIP")]
    print("  ".join(parts))


def write_json(results: list[Result], run_meta: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"

    payload = {
        **run_meta,
        "checks": [
            {
                "name": r.name,
                "status": r.status,
                "message": r.message,
                "metrics": r.metrics,
                "raw_output_file": r.raw_output_file,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_markdown(results: list[Result], run_meta: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.md"

    lines = [
        f"# gpuval run — {run_meta.get('run_id', '')}",
        "",
        f"- **SKU**: {run_meta.get('sku', '')}",
        f"- **Network mode**: {run_meta.get('network_mode', '')}",
        f"- **Nodes**: {', '.join(run_meta.get('nodes', ['localhost']))}",
        f"- **Profile**: {run_meta.get('profile', '')}",
        "",
        "| Check | Status | Message |",
        "|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r.name} | {r.status} | {r.message} |")

    lines.append("")

    for r in results:
        lines.append(f"## {r.name} — {r.status}")
        if r.metrics:
            lines.append("")
            lines.append("| Key | Value |")
            lines.append("|---|---|")
            for k, v in r.metrics.items():
                lines.append(f"| {k} | {v} |")
        if r.raw_output_file and Path(r.raw_output_file).exists():
            raw = Path(r.raw_output_file).read_text()
            lines.append("")
            lines.append("```")
            lines.append(textwrap.shorten(raw, width=4000, placeholder="\n...(truncated)"))
            lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines))
    return path
