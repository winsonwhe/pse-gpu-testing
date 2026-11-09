"""CLI entry point: list / run / report."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

# Ensure the repo root is on the path so gputest is importable when running
# as `python -m gputest.cli` or via the `gpuval` wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gputest.checks  # noqa: F401 — registers all checks as side-effect
from gputest.core import (
    Result,
    RunContext,
    get_registry,
    load_config,
    print_summary,
    run,
    run_checks,
    write_json,
    write_markdown,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _detect_network_mode(config: dict, ctx: RunContext | None = None) -> str:
    """Detect on target hosts through run(), respecting dry-run and mock mode."""
    cfg = config['network']
    if cfg.get('mode', 'auto') != 'auto':
        return cfg['mode']
    ctx = ctx or RunContext(config, 'gb300', [])
    command = [config['tools']['curl'], '-fsSL', '--max-time', str(cfg['imds_timeout_s']),
               '-H', 'Authorization: Bearer Oracle', cfg['imds_url']]
    results = ctx.run_parallel([{'cmd': command, 'node': node, 'timeout': cfg['imds_timeout_s'] + 5,
                                    'mock_fixture': 'imds_host.json'} for node in (ctx.nodes or [None])])
    modes = set()
    for rr in results:
        if rr.skipped:
            print('[dry-run] IMDS unavailable without execution; using network.fallback_mode. '
                  'Pass --network-mode to preview a specific mode.')
        try:
            data = json.loads(rr.stdout) if rr.returncode == 0 and not rr.skipped else None
            planes = data.get('rdmaFabricData', {}).get('planes') if data else None
            if data is not None and planes not in (None, 1, 4):
                raise ValueError(f'unsupported IMDS plane count: {planes}')
            modes.add(('multiplanar' if planes == 4 else 'single_plane') if data is not None
                      else cfg['fallback_mode'])
        except json.JSONDecodeError:
            modes.add(cfg['fallback_mode'])
    if len(modes) != 1:
        raise ValueError('mixed network modes across target hosts')
    return modes.pop()


def _resolve_checks(profile: str, tests_override: str | None, config: dict) -> list[str]:
    """Return the ordered list of check names to run."""
    if tests_override:
        return list(dict.fromkeys(t.strip() for t in tests_override.split(",") if t.strip()))
    profiles = config.get("profiles", {})
    if profile not in profiles:
        print(f"Unknown profile {profile!r}. Available: {', '.join(profiles)}", file=sys.stderr)
        sys.exit(1)
    return profiles[profile]


# ---------------------------------------------------------------------------
# sub-commands
# ---------------------------------------------------------------------------

def cmd_list(_args, config: dict) -> None:
    registry = get_registry()
    if not registry:
        print("No checks registered.")
        return
    w_name = max(len(c["name"]) for c in registry)
    w_cat = max(len(c["category"]) for c in registry)
    print(f"{'NAME':<{w_name}}  {'CATEGORY':<{w_cat}}  SKUs              MULTI-HOST")
    print("-" * (w_name + w_cat + 30))
    for c in registry:
        skus = ",".join(c["skus"]) if c["skus"] else "all"
        mh = "yes" if c["multi_host"] else "no"
        print(f"{c['name']:<{w_name}}  {c['category']:<{w_cat}}  {skus:<16}  {mh}")


def cmd_run(args, config: dict) -> None:
    if not args.sku:
        print("--sku is required (a10, gb300 or b300)", file=sys.stderr)
        sys.exit(1)

    known_skus = set(config.get("skus", {}).keys())
    if args.sku not in known_skus:
        print(f"Unknown SKU {args.sku!r}. Known: {', '.join(sorted(known_skus))}", file=sys.stderr)
        sys.exit(1)

    # --gpu-count overrides the SKU default (for partial A10 hosts: 1, 2, 3, or 4 GPUs)
    if getattr(args, "gpu_count", None) is not None:
        config.setdefault("skus", {})[args.sku]["gpu_count"] = args.gpu_count

    config["_profile"] = args.profile
    checks = _resolve_checks(args.profile, getattr(args, "tests", None), config)

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = Path(getattr(args, "output_dir", None) or (Path("results") / f"run-{run_id}"))

    nodes = []
    if getattr(args, "nodes", None):
        nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]

    ctx = RunContext(
        config=config,
        sku=args.sku,
        nodes=nodes,
        dry_run=getattr(args, "dry_run", False),
        mock=getattr(args, "mock", False),
        network_mode=config["network"]["fallback_mode"],
        baseline=getattr(args, "baseline", None),
        baseline_file=Path(getattr(args, "baseline_file", "baseline.json")),
        node_count=getattr(args, "node_count", None),
        results_dir=results_dir,
    )

    detection_error = None
    try:
        ctx.network_mode = _detect_network_mode(config, ctx)
    except (ValueError, AttributeError, TypeError) as exc:
        detection_error = str(exc)
    network_mode = ctx.network_mode

    print(f"gpuval run  sku={args.sku}  profile={args.profile}  "
          f"network={network_mode}  nodes={nodes or ['localhost']}")
    print(f"Results dir: {results_dir}")
    print()

    if detection_error:
        from gputest.reporting import annotate
        result = Result('network_mode', 'FAIL', detection_error)
        annotate(result, ctx)
        results = [result]
    else:
        results = run_checks(checks, ctx)

    print_summary(results)

    run_meta = {
        "run_id": run_id,
        "sku": args.sku,
        "profile": args.profile,
        "network_mode": network_mode,
        "nodes": nodes or ["localhost"],
        "burn_duration_s": ctx.burn_duration_s(),
        "dry_run": ctx.dry_run,
        "mock": ctx.mock,
        "config_file": str(Path(getattr(args, "config", "config.yaml")).resolve()),
    }
    jpath = write_json(results, run_meta, results_dir)
    mpath = write_markdown(results, run_meta, results_dir)
    print(f"\nJSON:     {jpath}")
    print(f"Markdown: {mpath}")
    return int(any(r.status == "FAIL" for r in results))


def cmd_setup_ssh(args, config: dict) -> None:
    """Provision keys using the shared bounded command helper; all writes honor dry-run."""
    nodes = [n.strip() for n in args.nodes.split(',') if n.strip()]
    if not nodes:
        raise ValueError('--nodes must contain at least one host')
    dry_run = args.dry_run
    tools = config['tools']
    ssh_dir = Path.home() / '.ssh'
    key = ssh_dir / 'id_ed25519'
    targets = [f'{args.user}@{n}' if args.user else n for n in nodes]
    failed = []

    def execute(cmd):
        return run(cmd, dry_run=dry_run, timeout=config['executor']['setup_timeout_s'],
                   results_dir=None if dry_run else Path('results/ssh-setup'))

    if not key.exists():
        if not dry_run:
            ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        rr = execute([tools['ssh_keygen'], '-t', 'ed25519', '-N', '', '-f', str(key)])
        if rr.returncode:
            print(rr.stderr)
            return 1
    for target in targets:
        rr = execute([tools['ssh_copy_id'], '-i', str(key) + '.pub', target])
        if rr.returncode:
            failed.append(target)
        scan = execute([tools['ssh_keyscan'], '-H', target.split('@')[-1]])
        fetched = execute([tools['ssh'], '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                           '--', target, 'cat ~/.ssh/id_ed25519.pub'])
        if scan.returncode or fetched.returncode or (not dry_run and not fetched.stdout.strip()):
            failed.append(target)
        if dry_run:
            print(f'[dry-run] append scanned host keys to {ssh_dir / "known_hosts"}')
            print(f'[dry-run] append remote public key to {ssh_dir / "authorized_keys"}')
            continue
        ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        for name, text in [('known_hosts', scan.stdout), ('authorized_keys', fetched.stdout)]:
            path = ssh_dir / name
            existing = path.read_text().splitlines() if path.exists() else []
            extra = [line for line in text.splitlines() if line and line not in existing]
            if extra:
                with path.open('a') as stream:
                    stream.write('\n'.join(extra) + '\n')
                path.chmod(0o600)
    print('Dry-run only: no keys changed.' if dry_run else f'SSH setup failures: {sorted(set(failed))}')
    print('Also provision launcher-to-peer SSH and trust host keys through your normal cluster process.')
    return int(bool(failed))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def cmd_report(args, _config: dict) -> None:
    result_dir = Path(args.result_dir)
    json_path = result_dir / "results.json"
    if not json_path.exists():
        print(f"No results.json in {result_dir}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(json_path.read_text())
    results = [
        Result(
            name=c["name"],
            status=c["status"],
            message=c.get("message", ""),
            metrics=c.get("metrics", {}),
            raw_output_file=c.get("raw_output_file"),
            thresholds=c.get("thresholds", []),
            hints=c.get("hints", []),
        )
        for c in data.get("checks", [])
    ]
    write_markdown(results, {k: v for k, v in data.items() if k != "checks"}, result_dir)
    print_summary(results)


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpuval",
        description="GPU validation toolkit for NVIDIA A10 / B300 / GB300 hosts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List all registered checks.")

    # run
    run_p = sub.add_parser("run", help="Run checks.")
    run_p.add_argument("--sku", required=True, choices=["a10", "gb300", "b300"],
                       help="Hardware SKU.")
    run_p.add_argument("--profile", default="standard",
                       choices=["quick", "standard", "full",
                                "a10_quick", "a10_standard", "a10_full"],
                       help="Check profile (default: standard).")
    run_p.add_argument("--gpu-count", type=_positive_int, default=None,
                       help="Override expected GPU count (useful for partial A10 hosts).")
    run_p.add_argument("--tests", metavar="CHECK,CHECK,...",
                       help="Override profile: run only these checks.")
    run_p.add_argument("--nodes", metavar="HOST,HOST,...",
                       help="Comma-separated SSH hosts (omit for localhost only).")
    run_p.add_argument("--node-count", type=_positive_int,
                       help="Select this many NCCL nodes from --nodes using GMF placement.")
    run_p.add_argument("--output-dir", help="Directory for JSON, Markdown and raw output.")
    run_p.add_argument("--baseline", choices=["save", "compare"],
                       help="Save or compare a known-good run and performance baselines.")
    run_p.add_argument("--baseline-file", default="baseline.json",
                       help="Baseline JSON path (default: baseline.json).")
    run_p.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing them.")
    run_p.add_argument("--mock", action="store_true",
                       help="Read outputs from fixtures/ instead of running commands.")
    run_p.add_argument("--network-mode", choices=["auto", "single_plane", "multiplanar"],
                       default=None,
                       help="Override network mode detection.")
    run_p.add_argument("--config", default="config.yaml",
                       help="Path to config.yaml (default: config.yaml).")

    # report
    rep_p = sub.add_parser("report", help="Re-render a saved result directory.")
    rep_p.add_argument("result_dir", help="Path to a results/run-<timestamp>/ directory.")

    # setup-ssh
    ssh_p = sub.add_parser(
        "setup-ssh",
        help="Configure passwordless SSH between this host and --nodes (required for NCCL).",
    )
    ssh_p.add_argument("--nodes", required=True, metavar="HOST,HOST,...",
                       help="Comma-separated list of remote hosts.")
    ssh_p.add_argument("--user", default=None,
                       help="Remote SSH user (default: current user).")
    ssh_p.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing them.")
    ssh_p.add_argument("--config", default="config.yaml",
                       help="Path to config.yaml (default: config.yaml).")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = getattr(args, "config", "config.yaml")
    config = load_config(config_path)

    # CLI override for network mode
    if getattr(args, "network_mode", None):
        config.setdefault("network", {})["mode"] = args.network_mode

    dispatch = {
        "list": cmd_list,
        "run": cmd_run,
        "report": cmd_report,
        "setup-ssh": cmd_setup_ssh,
    }
    status = dispatch[args.command](args, config)
    if status:
        sys.exit(status)


if __name__ == "__main__":
    main()
