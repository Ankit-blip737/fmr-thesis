"""Commit + push result artifacts back to origin (used by the Colab notebooks).

Auth: expects a GitHub token in env ``GH_TOKEN`` (Instance A) — see BLOCKERS.md.
Pushes to the current branch (``master`` for Instance A). Safe to import on the
dev box; ``push`` is only invoked on Colab.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = "github.com/Ankit-blip737/fmr-thesis.git"


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def push(paths: str | list[str], message: str, branch: str = "master") -> None:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    paths = [paths] if isinstance(paths, str) else list(paths)

    # Identity (Colab has none by default).
    _run(["git", "config", "user.email", "fmr-bot@colab"])
    _run(["git", "config", "user.name", "FMR Colab Runner (A)"])

    _run(["git", "add", *paths, "RESULTS_LOG.md", "BLOCKERS.md", "DECISIONS.md"])
    status = _run(["git", "status", "--porcelain"])
    if not status:
        print("[push] nothing to commit")
        return
    _run(["git", "commit", "-m", message])

    if token:
        remote = f"https://x-access-token:{token}@{REPO}"
        _run(["git", "push", remote, f"HEAD:{branch}"])
    else:
        _run(["git", "push", "origin", f"HEAD:{branch}"])
    print(f"[push] pushed to {branch}: {message}")


if __name__ == "__main__":
    import sys

    push(sys.argv[1] if len(sys.argv) > 1 else "fmr/outputs/real",
         message=sys.argv[2] if len(sys.argv) > 2 else "[A] real results")
