"""Consulta del estado de Git para la trazabilidad de los runs."""

import subprocess

from dsrv.config import REPO_ROOT


class DirtyWorktreeError(RuntimeError):
    """El árbol de trabajo tiene cambios sin commit."""


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def get_git_commit() -> str:
    """Devuelve el SHA completo de HEAD."""
    return _git("rev-parse", "HEAD")


def assert_clean_worktree() -> None:
    """Lanza DirtyWorktreeError si `git status --porcelain` no está vacío."""
    status = _git("status", "--porcelain")
    if status:
        raise DirtyWorktreeError(f"Haz commit antes de continuar. Cambios pendientes:\n{status}")


def get_git_tag_or_none() -> str | None:
    """Devuelve el tag más reciente que apunta a HEAD, o None si no hay ninguno."""
    tags = _git("tag", "--points-at", "HEAD", "--sort=-creatordate").splitlines()
    return tags[0] if tags else None
