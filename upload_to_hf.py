"""
upload_to_hf.py — Deploy the Data Cleaning OpenEnv to HuggingFace Spaces.

Primary method: openenv CLI (recommended)
    openenv push --repo-id YOUR_USERNAME/data-cleaning-openenv

Fallback method: this script (uses huggingface_hub directly)
    pip install huggingface_hub
    python upload_to_hf.py --repo YOUR_USERNAME/data-cleaning-openenv --token hf_xxx

    # Or set token via env var:
    export HF_TOKEN=hf_xxx
    python upload_to_hf.py --repo YOUR_USERNAME/data-cleaning-openenv

    # Dry run:
    python upload_to_hf.py --repo YOUR_USERNAME/data-cleaning-openenv --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).parent

UPLOAD_FILES = [
    "README.md",
    "openenv.yaml",
    "pyproject.toml",
    "requirements.txt",
    "Dockerfile",
    "models.py",
    "client.py",
    "baseline.py",
    "train_grpo.py",
    "__init__.py",
    "server/__init__.py",
    "server/app.py",
    "server/dc_environment.py",
    "server/environment.py",
    "tests/__init__.py",
    "tests/test_units.py",
]


def try_openenv_cli(repo_id: str, private: bool) -> bool:
    """Attempt to push using the openenv CLI. Returns True if successful."""
    import shutil
    if not shutil.which("openenv"):
        return False
    import subprocess
    args = ["openenv", "push", "--repo-id", repo_id]
    if private:
        args.append("--private")
    print(f"  Running: {' '.join(args)}")
    result = subprocess.run(args, cwd=str(ROOT))
    return result.returncode == 0


def push_via_hub(repo_id: str, token: str, private: bool, dry_run: bool) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=token)

    if not dry_run:
        api.create_repo(
            repo_id=repo_id, repo_type="space",
            space_sdk="docker", private=private, exist_ok=True,
        )
        print(f"  + Space: https://huggingface.co/spaces/{repo_id}")

    print(f"\n  Uploading {len(UPLOAD_FILES)} files...\n")
    ok = failed = 0

    for rel in UPLOAD_FILES:
        local = ROOT / rel
        if not local.exists():
            print(f"  ! skip  {rel}")
            continue
        if dry_run:
            print(f"  ~ {rel}")
            ok += 1
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=rel,
                repo_id=repo_id,
                repo_type="space",
            )
            print(f"  + {rel}")
            ok += 1
        except Exception as e:
            print(f"  x {rel}: {e}")
            failed += 1

    print(f"\n  {ok} uploaded, {failed} failed")

    if not dry_run:
        url = f"https://huggingface.co/spaces/{repo_id}"
        print(f"""
  Space URL: {url}
  API:       {url}/health

  After the Docker build completes (~2-3 min):
    from client import DataCleaningEnv
    with DataCleaningEnv(base_url="{url}").sync() as env:
        result = env.reset(task_id="ecommerce_easy", seed=42)
        print(result.observation.df_preview)
""")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Deploy Data Cleaning OpenEnv to HuggingFace Spaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Preferred: openenv push --repo-id YOUR_USERNAME/data-cleaning-openenv
            Fallback:  python upload_to_hf.py --repo YOUR_USERNAME/data-cleaning-openenv
        """),
    )
    p.add_argument("--repo",    required=True, help="HF repo: username/name")
    p.add_argument("--token",   default=os.environ.get("HF_TOKEN"),
                   help="HF write token (or set HF_TOKEN)")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-cli", action="store_true",
                   help="Skip openenv CLI attempt, go straight to huggingface_hub")
    args = p.parse_args()

    print(f"\n  Data Cleaning OpenEnv → HuggingFace Spaces")
    print(f"  repo: {args.repo}  |  dry-run: {args.dry_run}\n")

    # Try openenv CLI first (the canonical way)
    if not args.dry_run and not args.skip_cli:
        print("  Trying openenv CLI...")
        if try_openenv_cli(args.repo, args.private):
            print("  ✓ Deployed via openenv CLI")
            return
        print("  openenv CLI not found or failed — falling back to huggingface_hub\n")

    # Fallback: huggingface_hub
    if not args.dry_run and not args.token:
        print("ERROR: --token required (or set HF_TOKEN=...)")
        print("  Get a token: https://huggingface.co/settings/tokens")
        sys.exit(1)

    try:
        import huggingface_hub  # noqa
    except ImportError:
        print("ERROR: pip install huggingface_hub")
        sys.exit(1)

    push_via_hub(args.repo, args.token, args.private, args.dry_run)


if __name__ == "__main__":
    main()