"""CLI entry point: list / run / report."""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
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
    run_checks,
    write_json,
    write_markdown,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _detect_network_mode(config: dict) -> str:
    """Detect single_plane vs multiplanar; fall back to config default."""
    mode = config.get("network", {}).get("mode", "auto")
    if mode != "auto":
        return mode

    imds_url = config["network"].get("imds_url", "http://169.254.169.254/opc/v2/host")
    timeout = config["network"].get("imds_timeout_s", 2)
    try:
        import urllib.request
        req = urllib.request.Request(imds_url, headers={"Authorization": "Bearer Oracle"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        planes = data.get("rdmaFabricData", {}).get("planes")
        if planes == 4:
            return "multiplanar"
        return "single_plane"
    except (OSError, ValueError, KeyError):
        return "single_plane"


def _resolve_checks(profile: str, tests_override: str | None, config: dict) -> list[str]:
    """Return the ordered list of check names to run."""
    if tests_override:
        return [t.strip() for t in tests_override.split(",") if t.strip()]
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

    checks = _resolve_checks(args.profile, getattr(args, "tests", None), config)

    network_mode = _detect_network_mode(config)

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = Path("results") / f"run-{run_id}"

    nodes = []
    if getattr(args, "nodes", None):
        nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]

    ctx = RunContext(
        config=config,
        sku=args.sku,
        nodes=nodes,
        dry_run=getattr(args, "dry_run", False),
        mock=getattr(args, "mock", False),
        network_mode=network_mode,
        results_dir=results_dir,
    )

    print(f"gpuval run  sku={args.sku}  profile={args.profile}  "
          f"network={network_mode}  nodes={nodes or ['localhost']}")
    print(f"Results dir: {results_dir}")
    print()

    results = run_checks(checks, ctx)

    print_summary(results)

    run_meta = {
        "run_id": run_id,
        "sku": args.sku,
        "profile": args.profile,
        "network_mode": network_mode,
        "nodes": nodes or ["localhost"],
    }
    jpath = write_json(results, run_meta, results_dir)
    mpath = write_markdown(results, run_meta, results_dir)
    print(f"\nJSON:     {jpath}")
    print(f"Markdown: {mpath}")


def cmd_setup_ssh(args, _config: dict) -> None:
    """
    Configure passwordless SSH between this host and every node in --nodes.

    Steps performed (with --dry-run support):
      1. Generate an Ed25519 key at ~/.ssh/id_ed25519 if none exists.
      2. ssh-copy-id to each remote node (prompts once for the remote password).
      3. From each remote node, copy this host's key back (reverse direction for MPI).
      4. Pre-populate ~/.ssh/known_hosts for all nodes so BatchMode doesn't block.

    After this command, run:
      gpuval run --sku <sku> --profile <profile> --nodes <nodes>
    """
    if not args.nodes:
        print("--nodes HOST,HOST,... is required for setup-ssh", file=sys.stderr)
        sys.exit(1)

    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    dry_run = getattr(args, "dry_run", False)
    user = getattr(args, "user", None) or ""
    key_path = Path.home() / ".ssh" / "id_ed25519"
    pub_key_path = key_path.with_suffix(".pub")

    def _run(cmd: list[str], desc: str) -> bool:
        if dry_run:
            print(f"[dry-run] {' '.join(cmd)}")
            return True
        print(f"  {desc} ... ", end="", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("OK")
            return True
        print(f"FAILED\n    {result.stderr.strip()[:200]}")
        return False

    print("=== gpuval setup-ssh ===")
    print(f"Nodes: {nodes}")
    print()

    # Step 1: generate key
    if not key_path.exists():
        print("[1/4] Generating Ed25519 SSH key")
        ok = _run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key_path)],
            f"ssh-keygen → {key_path}",
        )
        if not ok and not dry_run:
            sys.exit(1)
    else:
        print(f"[1/4] Key exists: {key_path} (skipping generation)")

    print()

    # Step 2: copy public key to each node (forward direction)
    print("[2/4] Copying public key to remote nodes (you may be prompted for passwords)")
    forward_failed = []
    for node in nodes:
        target = f"{user}@{node}" if user else node
        ok = _run(
            ["ssh-copy-id", "-i", str(pub_key_path), "-o", "StrictHostKeyChecking=no", target],
            f"ssh-copy-id → {target}",
        )
        if not ok:
            forward_failed.append(node)

    print()

    # Step 3: pre-populate known_hosts for all nodes (avoids BatchMode failing on first connect)
    print("[3/4] Pre-populating ~/.ssh/known_hosts")
    known = Path.home() / ".ssh" / "known_hosts"
    for node in nodes:
        if dry_run:
            print(f"[dry-run] ssh-keyscan -H {node} >> {known}")
            continue
        scan = subprocess.run(
            ["ssh-keyscan", "-H", node], capture_output=True, text=True, check=False,
        )
        if scan.returncode == 0 and scan.stdout.strip():
            known.parent.mkdir(mode=0o700, exist_ok=True)
            with open(known, "a") as f:
                f.write(scan.stdout)
            print(f"  ssh-keyscan {node} ... OK")
        else:
            print(f"  ssh-keyscan {node} ... FAILED")

    print()

    # Step 4: reverse direction — each node's public key goes into this host's authorized_keys
    local_auth = Path.home() / ".ssh" / "authorized_keys"
    print(f"[4/4] Adding each node's public key to {local_auth}")
    reverse_failed = []
    for node in nodes:
        target = f"{user}@{node}" if user else node
        if dry_run:
            print(f"[dry-run] ssh {target} cat ~/.ssh/id_ed25519.pub  >> {local_auth}")
            continue
        fetched = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target,
             "cat ~/.ssh/id_ed25519.pub"],
            capture_output=True, text=True, check=False,
        )
        pub = fetched.stdout.strip()
        if fetched.returncode != 0 or not pub:
            print(f"  {target}: no public key readable (run ssh-keygen on it first)")
            reverse_failed.append(node)
            continue
        existing = local_auth.read_text().splitlines() if local_auth.exists() else []
        if pub not in existing:
            local_auth.parent.mkdir(mode=0o700, exist_ok=True)
            with open(local_auth, "a") as f:
                f.write(pub + "\n")
            local_auth.chmod(0o600)
            print(f"  {target}: added")
        else:
            print(f"  {target}: already present")

    print()
    print("=== Summary ===")
    if forward_failed:
        print(f"  FORWARD failed (this → node): {forward_failed}")
        print("  Fix: ensure you can SSH to these hosts and re-run setup-ssh")
    else:
        print("  FORWARD (this → nodes): OK")

    if reverse_failed:
        print(f"  REVERSE failed (node → this): {reverse_failed}")
        print("  Fix: on each failed node run ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519, then re-run")
    else:
        print("  REVERSE (nodes → this): OK")

    print()
    if dry_run:
        print("Dry-run only: nothing was changed. Re-run without --dry-run to apply.")
    elif not forward_failed and not reverse_failed:
        print("Passwordless SSH configured. Verify with:")
        print(f"  gpuval run --sku <sku> --profile quick --tests ssh_connectivity "
              f"--nodes {','.join(nodes)}")
    else:
        print("Some steps failed — fix the issues above and re-run setup-ssh.")
        sys.exit(1)


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
        )
        for c in data.get("checks", [])
    ]
    print_summary(results)


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
    run_p.add_argument("--gpu-count", type=int, default=None,
                       help="Override expected GPU count (useful for partial A10 hosts).")
    run_p.add_argument("--tests", metavar="CHECK,CHECK,...",
                       help="Override profile: run only these checks.")
    run_p.add_argument("--nodes", metavar="HOST,HOST,...",
                       help="Comma-separated SSH hosts (omit for localhost only).")
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
    dispatch[args.command](args, config)


if __name__ == "__main__":
    main()
