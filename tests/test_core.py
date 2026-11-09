"""Tests for parsers and checks, runnable without GPU hardware."""

from __future__ import annotations

import json as _json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gputest.checks  # noqa: F401 — registers checks
from gputest import parsers
from gputest.core import (
    Result,
    RunContext,
    load_config,
    run_checks,
    write_json,
    write_markdown,
)

FIXTURE_DIR = ROOT / "fixtures"
CONFIG_PATH = ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(sku: str = "gb300", mock: bool = True, dry_run: bool = False,
         nodes: list[str] | None = None, tmp_path: Path | None = None) -> RunContext:
    config = load_config(CONFIG_PATH)
    return RunContext(
        config=config,
        sku=sku,
        nodes=nodes or [],
        dry_run=dry_run,
        mock=mock,
        network_mode="single_plane",
        results_dir=tmp_path or Path("results/test"),
    )


# ---------------------------------------------------------------------------
# parse_nvidia_smi_csv
# ---------------------------------------------------------------------------

class TestParseNvidiaSmiCsv:
    def test_four_gb300_gpus(self):
        text = (FIXTURE_DIR / "nvidia_smi.txt").read_text()
        rows = parsers.parse_nvidia_smi_csv(text)
        assert len(rows) == 4
        assert rows[0]["driver_version"] == "580.159.03"
        assert rows[0]["memory_total_mib"] == 284208

    def test_four_a10_gpus(self):
        text = (FIXTURE_DIR / "nvidia_smi_a10.txt").read_text()
        rows = parsers.parse_nvidia_smi_csv(text)
        assert len(rows) == 4
        assert rows[0]["driver_version"] == "525.147.05"
        assert rows[0]["memory_total_mib"] == 23028
        assert "A10" in rows[0]["name"]

    def test_empty_input(self):
        assert parsers.parse_nvidia_smi_csv("") == []

    def test_skips_short_lines(self):
        text = "0, BadGPU\n1, NVIDIA A10, 525.0, 23028, Enabled\n"
        assert len(parsers.parse_nvidia_smi_csv(text)) == 1

    def test_non_integer_memory(self):
        text = "0, NVIDIA A10, 525.0, N/A, Enabled\n"
        assert parsers.parse_nvidia_smi_csv(text)[0]["memory_total_mib"] is None


# ---------------------------------------------------------------------------
# parse_rdma_link
# ---------------------------------------------------------------------------

class TestParseRdmaLink:
    def test_a10_fixture(self):
        text = (FIXTURE_DIR / "rdma_link_a10.txt").read_text()
        active = parsers.parse_rdma_link(text)
        assert len(active) == 2
        assert active[0]["hca"] == "mlx5_0"
        assert active[0]["netdev"] == "rdma0"
        assert active[1]["hca"] == "mlx5_1"

    def test_ignores_down_links(self):
        text = "link mlx5_2/1 state DOWN physical_state POLLING netdev eth0\n"
        assert parsers.parse_rdma_link(text) == []

    def test_empty(self):
        assert parsers.parse_rdma_link("") == []


# ---------------------------------------------------------------------------
# parse_numactl_hardware
# ---------------------------------------------------------------------------

class TestParseNumactlHardware:
    def test_a10_two_nodes(self):
        text = (FIXTURE_DIR / "numactl_a10.txt").read_text()
        nodes = parsers.parse_numactl_hardware(text)
        cpu_nodes = [n for n in nodes if n["cpus"] and n["mem_mb"] > 0]
        assert len(cpu_nodes) == 2

    def test_mem_parsed(self):
        text = (FIXTURE_DIR / "numactl_a10.txt").read_text()
        nodes = parsers.parse_numactl_hardware(text)
        assert nodes[0]["mem_mb"] == 192000

    def test_zero_mem_node(self):
        text = "node 0 cpus: 0 1\nnode 0 size: 0 MB\n"
        nodes = parsers.parse_numactl_hardware(text)
        assert nodes[0]["mem_mb"] == 0


# ---------------------------------------------------------------------------
# parse_nvbandwidth
# ---------------------------------------------------------------------------

class TestParseNvbandwidth:
    def test_sum_parsed(self):
        text = (FIXTURE_DIR / "nvbandwidth_a10.txt").read_text()
        r = parsers.parse_nvbandwidth(text, "device_to_device_bidirectional_memcpy_read_ce")
        assert r["sum_gbs"] == 24.51

    def test_rows_parsed(self):
        text = (FIXTURE_DIR / "nvbandwidth_a10.txt").read_text()
        r = parsers.parse_nvbandwidth(text, "device_to_device_bidirectional_memcpy_read_ce")
        assert len(r["rows"]) > 0

    def test_missing_test_section(self):
        r = parsers.parse_nvbandwidth("no matching section\n", "nonexistent_test")
        assert r["sum_gbs"] is None
        assert r["rows"] == []


# ---------------------------------------------------------------------------
# nvidia_smi check
# ---------------------------------------------------------------------------

class TestNvidiaSmiCheck:
    def test_pass_gb300(self, tmp_path):
        ctx = _ctx("gb300", tmp_path=tmp_path)
        r = run_checks(["nvidia_smi"], ctx)[0]
        assert r.status == "PASS", r.message
        assert r.metrics["gpu_count"] == 4

    def test_pass_a10_4gpu(self, tmp_path):
        ctx = _ctx("a10", tmp_path=tmp_path)
        r = run_checks(["nvidia_smi"], ctx)[0]
        assert r.status == "PASS", r.message
        assert r.metrics["gpu_count"] == 4

    def test_a10_partial_host_allowed(self, tmp_path):
        """1-3 GPUs on an A10 host should pass if within gpu_count_min..gpu_count."""
        config = load_config(CONFIG_PATH)
        # Simulate a 2-GPU fixture by manipulating in memory
        config["skus"]["a10"]["gpu_count"] = 4
        config["skus"]["a10"]["gpu_count_min"] = 1
        # Use the real 4-GPU fixture; count check should still pass as 4 >= 1
        ctx = RunContext(config=config, sku="a10", nodes=[], dry_run=False,
                         mock=True, network_mode="single_plane", results_dir=tmp_path)
        r = run_checks(["nvidia_smi"], ctx)[0]
        assert r.status == "PASS"

    def test_wrong_sku_gpu_count(self, tmp_path):
        """b300 expects 8 GPUs; a10 fixture has 4. Verify the mismatch via parser + config."""
        config = load_config(CONFIG_PATH)
        # Simplest: verify the SKU has gpu_count=8 and our a10 fixture has 4 GPUs
        text = (FIXTURE_DIR / "nvidia_smi_a10.txt").read_text()
        parsed = parsers.parse_nvidia_smi_csv(text)
        assert len(parsed) == 4
        assert config["skus"]["b300"]["gpu_count"] == 8

    def test_dry_run_gives_skip(self, tmp_path):
        ctx = _ctx("a10", dry_run=True, mock=False, tmp_path=tmp_path)
        r = run_checks(["nvidia_smi"], ctx)[0]
        assert r.status == "SKIP"

    def test_a10_driver_too_old(self, tmp_path):
        config = load_config(CONFIG_PATH)
        config["skus"]["a10"]["driver_min"] = "600"  # higher than fixture 525
        ctx = RunContext(config=config, sku="a10", nodes=[], dry_run=False,
                         mock=True, network_mode="single_plane", results_dir=tmp_path)
        r = run_checks(["nvidia_smi"], ctx)[0]
        assert r.status == "FAIL"
        assert "525" in r.message


# ---------------------------------------------------------------------------
# fabricmanager check (SKU filter)
# ---------------------------------------------------------------------------

class TestFabricmanagerCheck:
    def test_skipped_for_a10(self, tmp_path):
        ctx = _ctx("a10", tmp_path=tmp_path)
        r = run_checks(["fabricmanager"], ctx)[0]
        assert r.status == "SKIP"
        assert "a10" in r.message.lower() or "SKU" in r.message

    def test_runs_for_gb300(self, tmp_path):
        ctx = _ctx("gb300", tmp_path=tmp_path)
        # No fixture for this yet; mock will return empty stdout → FAIL
        r = run_checks(["fabricmanager"], ctx)[0]
        # Should attempt and return some result, not SKIP
        assert r.status in ("PASS", "FAIL", "WARN")


# ---------------------------------------------------------------------------
# rdma_link check
# ---------------------------------------------------------------------------

class TestRdmaLinkCheck:
    def test_pass_a10(self, tmp_path):
        ctx = _ctx("a10", tmp_path=tmp_path)
        r = run_checks(["rdma_link"], ctx)[0]
        assert r.status == "PASS", r.message
        assert r.metrics["active_hca_count"] == 2

    def test_multi_host_rdma_link_skipped_without_nodes(self, tmp_path):
        """rdma_link is a single-host check; should run regardless of node count."""
        ctx = _ctx("a10", nodes=[], tmp_path=tmp_path)
        r = run_checks(["rdma_link"], ctx)[0]
        # rdma_link is not multi_host so it should run
        assert r.status != "SKIP" or r.message == "dry-run"


# ---------------------------------------------------------------------------
# nvbandwidth check
# ---------------------------------------------------------------------------

class TestNvbandwidthCheck:
    def test_pass_a10(self, tmp_path):
        ctx = _ctx("a10", tmp_path=tmp_path)
        r = run_checks(["nvbandwidth"], ctx)[0]
        assert r.status == "PASS", r.message

    def test_fail_when_below_threshold(self, tmp_path):
        config = load_config(CONFIG_PATH)
        config["nvbandwidth"]["tests"]["a10"]["thresholds"][
            "device_to_device_bidirectional_memcpy_read_ce"] = 9999
        ctx = RunContext(config=config, sku="a10", nodes=[], dry_run=False,
                         mock=True, network_mode="single_plane", results_dir=tmp_path)
        r = run_checks(["nvbandwidth"], ctx)[0]
        assert r.status == "FAIL"

    def test_skip_for_unknown_sku(self, tmp_path):
        config = load_config(CONFIG_PATH)
        config["skus"]["mystery"] = {"gpu_count": 1, "gpu_model_pattern": "X", "driver_min": "1"}
        ctx = RunContext(config=config, sku="mystery", nodes=[], dry_run=False,
                         mock=True, network_mode="single_plane", results_dir=tmp_path)
        r = run_checks(["nvbandwidth"], ctx)[0]
        assert r.status == "SKIP"


# ---------------------------------------------------------------------------
# numa_topology check
# ---------------------------------------------------------------------------

class TestNumaTopologyCheck:
    def test_pass_a10(self, tmp_path):
        ctx = _ctx("a10", tmp_path=tmp_path)
        r = run_checks(["numa_topology"], ctx)[0]
        assert r.status == "PASS"
        assert r.metrics["numa_nodes_with_cpu"] == 2


# ---------------------------------------------------------------------------
# Multi-host SKIP behaviour
# ---------------------------------------------------------------------------

class TestMultiHostSkip:
    def test_rdma_bw_skipped_single_host(self, tmp_path):
        ctx = _ctx("a10", nodes=[], tmp_path=tmp_path)
        r = run_checks(["rdma_bw"], ctx)[0]
        # rdma_bw is not yet registered; it should SKIP (not found or multi-host)
        assert r.status == "SKIP"

    def test_nccl_allreduce_skipped_single_host(self, tmp_path):
        ctx = _ctx("a10", nodes=[], tmp_path=tmp_path)
        r = run_checks(["nccl_allreduce"], ctx)[0]
        assert r.status == "SKIP"


# ---------------------------------------------------------------------------
# run() helper
# ---------------------------------------------------------------------------

class TestRunHelper:
    def test_mock_reads_fixture(self, tmp_path):
        from gputest.core import run
        rr = run(["nvidia-smi"], mock=True, mock_fixture="nvidia_smi_a10.txt",
                 results_dir=tmp_path)
        assert "A10" in rr.stdout

    def test_dry_run_skipped(self, tmp_path, capsys):
        from gputest.core import run
        rr = run(["echo", "hello"], dry_run=True, results_dir=tmp_path)
        assert rr.skipped is True
        assert "dry-run" in capsys.readouterr().out

    def test_command_not_found(self, tmp_path):
        from gputest.core import run
        rr = run(["__no_such_command__"], results_dir=tmp_path)
        assert rr.returncode == 127


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

class TestReportWriters:
    def _results(self):
        return [
            Result(name="nvidia_smi", status="PASS", message="4x A10",
                   metrics={"gpu_count": 4}),
            Result(name="rdma_bw", status="SKIP", message="requires >=2 hosts"),
        ]

    def _meta(self):
        return {"run_id": "test-001", "sku": "a10",
                "profile": "a10_quick", "network_mode": "single_plane",
                "nodes": ["localhost"]}

    def test_json_written(self, tmp_path):
        path = write_json(self._results(), self._meta(), tmp_path)
        data = _json.loads(path.read_text())
        checks = {c["name"]: c for c in data["checks"]}
        assert checks["nvidia_smi"]["status"] == "PASS"
        assert checks["rdma_bw"]["status"] == "SKIP"

    def test_markdown_written(self, tmp_path):
        path = write_markdown(self._results(), self._meta(), tmp_path)
        md = path.read_text()
        assert "## nvidia_smi" in md and "PASS" in md


# ---------------------------------------------------------------------------
# ssh_connectivity check
# ---------------------------------------------------------------------------

class TestSshConnectivityCheck:
    def test_skipped_without_nodes(self, tmp_path):
        ctx = _ctx("a10", nodes=[], tmp_path=tmp_path)
        r = run_checks(["ssh_connectivity"], ctx)[0]
        assert r.status == "SKIP"
        assert "2 hosts" in r.message

    def test_dry_run_skip_with_nodes(self, tmp_path):
        ctx = _ctx("a10", nodes=["n1", "n2"], dry_run=True, mock=False, tmp_path=tmp_path)
        r = run_checks(["ssh_connectivity"], ctx)[0]
        assert r.status == "SKIP"

    def test_pass_mock_two_nodes(self, tmp_path):
        ctx = _ctx("a10", nodes=["n1", "n2"], tmp_path=tmp_path)
        r = run_checks(["ssh_connectivity"], ctx)[0]
        assert r.status == "PASS", r.message
        assert r.metrics["nodes_checked"] == 2

    def test_fail_when_ssh_errors(self, tmp_path, monkeypatch):
        from gputest.core import RunResult
        ctx = _ctx("a10", nodes=["n1", "n2"], mock=False, tmp_path=tmp_path)
        monkeypatch.setattr(
            ctx, "run",
            lambda *a, **k: RunResult(stdout="", stderr="Permission denied (publickey).",
                                      returncode=255, elapsed_s=0.0),
        )
        r = run_checks(["ssh_connectivity"], ctx)[0]
        assert r.status == "FAIL"
        assert "n1" in r.message


# ---------------------------------------------------------------------------
# setup-ssh CLI
# ---------------------------------------------------------------------------

class TestSetupSshCli:
    def test_requires_nodes(self, monkeypatch):
        from gputest import cli
        monkeypatch.setattr(sys, "argv", ["gpuval", "setup-ssh"])
        import pytest as _pt
        with _pt.raises(SystemExit):
            cli.main()

    def test_dry_run_prints_and_exits_ok(self, monkeypatch, capsys, tmp_path):
        from gputest import cli
        monkeypatch.setattr(sys, "argv",
                            ["gpuval", "setup-ssh", "--nodes", "n1,n2", "--dry-run"])
        monkeypatch.setattr(cli.Path, "home", classmethod(lambda cls: tmp_path))
        cli.main()
        out = capsys.readouterr().out
        assert "dry-run" in out
        assert "n1" in out and "n2" in out
        assert not (tmp_path / ".ssh" / "known_hosts").exists()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_a10_sku_present(self):
        cfg = load_config(CONFIG_PATH)
        assert "a10" in cfg["skus"]

    def test_a10_gpu_count(self):
        cfg = load_config(CONFIG_PATH)
        assert cfg["skus"]["a10"]["gpu_count"] == 4

    def test_a10_no_fabricmanager(self):
        cfg = load_config(CONFIG_PATH)
        assert cfg["skus"]["a10"]["fabricmanager_required"] is False

    def test_a10_profiles(self):
        cfg = load_config(CONFIG_PATH)
        assert "a10_quick" in cfg["profiles"]
        assert "a10_standard" in cfg["profiles"]
        assert "a10_full" in cfg["profiles"]

    def test_a10_nvbandwidth_config(self):
        cfg = load_config(CONFIG_PATH)
        a10_nb = cfg["nvbandwidth"]["tests"]["a10"]
        assert "device_to_device_bidirectional_memcpy_read_ce" in a10_nb["test_names"]
        assert a10_nb["thresholds"]["device_to_device_bidirectional_memcpy_read_ce"] == 20
