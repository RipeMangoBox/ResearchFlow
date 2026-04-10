#!/usr/bin/env python3
"""Create cross-platform skill aliases for Claude Code and Codex.

The single source of truth stays in `.claude/skills` and
`.claude/skills-config.json`.

This script creates compatibility aliases under:

- `.codex/skills`
- `.codex/skills-config.json`

On macOS/Linux it uses symlinks.
On Windows it uses directory junctions for folders and hard links for the
config file so the same skill library is shared without copying.
"""

from __future__ import annotations

import argparse
import os
import platform
import stat as statmod
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_reparse_point(path: Path) -> bool:
    try:
        attrs = os.lstat(path).st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    flag = getattr(statmod, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attrs & flag)


def path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def points_to(target: Path, source: Path) -> bool:
    if not path_exists(target):
        return False
    try:
        if os.path.samefile(target, source):
            return True
    except OSError:
        pass
    try:
        return target.resolve() == source.resolve()
    except OSError:
        return False


def remove_existing(target: Path, source: Path) -> None:
    if not path_exists(target):
        return
    if points_to(target, source):
        return
    if target.is_symlink() or target.is_file():
        target.unlink()
        return
    if target.is_dir():
        if is_windows() and is_reparse_point(target):
            target.rmdir()
            return
        if any(target.iterdir()):
            raise RuntimeError(
                f"{target} already exists as a real non-empty directory. "
                "Please move it away before running this script."
            )
        target.rmdir()
        return
    raise RuntimeError(f"Unsupported existing path type: {target}")


def create_dir_alias(source: Path, target: Path) -> None:
    if path_exists(target):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_windows():
        command = f'mklink /J "{target}" "{source}"'
        subprocess.run(["cmd", "/c", command], check=True)
        return
    relative_source = os.path.relpath(source, start=target.parent)
    target.symlink_to(relative_source, target_is_directory=True)


def create_file_alias(source: Path, target: Path) -> None:
    if path_exists(target):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_windows():
        os.link(source, target)
        return
    relative_source = os.path.relpath(source, start=target.parent)
    target.symlink_to(relative_source)


def remove_path_if_present(path: Path) -> None:
    if not path_exists(path):
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        if is_windows() and is_reparse_point(path):
            path.rmdir()
            return
        if any(path.iterdir()):
            raise RuntimeError(
                f"{path} already exists as a real non-empty directory. "
                "Please move it away before running this script."
            )
        path.rmdir()
        return
    raise RuntimeError(f"Unsupported existing path type: {path}")


def cleanup_legacy_agents_aliases(root: Path) -> None:
    legacy_paths = [
        root / ".agents" / "skills",
        root / ".agents" / "skills-config.json",
    ]
    for path in legacy_paths:
        remove_path_if_present(path)
    remove_path_if_present(root / ".agents")


def alias_pairs() -> list[tuple[Path, Path]]:
    root = repo_root()
    return [
        (root / ".codex" / "skills", root / ".claude" / "skills"),
        (root / ".codex" / "skills-config.json", root / ".claude" / "skills-config.json"),
    ]


def install_aliases() -> list[tuple[Path, Path]]:
    root = repo_root()
    cleanup_legacy_agents_aliases(root)
    pairs = alias_pairs()
    for target, source in pairs:
        if target.name == "skills":
            if not source.is_dir():
                raise RuntimeError(f"Missing source skills directory: {source}")
        else:
            if not source.is_file():
                raise RuntimeError(f"Missing source config file: {source}")
        remove_existing(target, source)
        if source.is_dir():
            create_dir_alias(source, target)
        else:
            create_file_alias(source, target)
    return pairs


def check_aliases() -> int:
    missing = []
    for target, source in alias_pairs():
        if not points_to(target, source):
            missing.append((target, source))
    if missing:
        for target, source in missing:
            print(f"missing: {target} -> {source}")
        return 1
    print("shared skill aliases are ready")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create shared Claude/Codex skill aliases for this repository."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the shared-skill aliases exist after setup",
    )
    args = parser.parse_args()

    try:
        pairs = install_aliases()
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        return 1

    root = repo_root()
    for target, source in pairs:
        print(f"{target.relative_to(root)} -> {source.relative_to(root)}")

    if args.check:
        return check_aliases()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
