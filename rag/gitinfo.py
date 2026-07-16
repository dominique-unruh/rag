"""Git revision / clean-tree checks (PLAN §5 git gating).

Translations may only be committed to the DB when the working tree is clean, so
the recorded ``git_rev`` is meaningful. These helpers wrap the few git commands
needed for that check.
"""

from __future__ import annotations

import subprocess

from .errors import RagError


class GitError(RagError):
    """Raised when a git command fails (e.g. not a git repository)."""


def git_revision() -> str:
    """Return the current ``HEAD`` commit hash.

    Raises :class:`GitError` if git is unavailable or this is not a repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GitError(f"could not determine git revision: {exc}") from exc
    return result.stdout.strip()


def is_clean() -> bool:
    """Return whether the working tree has no uncommitted changes.

    Uses ``git status --porcelain``: empty output means a clean tree. Raises
    :class:`GitError` if git fails.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GitError(f"could not check git status: {exc}") from exc
    # Any porcelain line means a modified/untracked file — i.e. a dirty tree.
    return result.stdout.strip() == ""
