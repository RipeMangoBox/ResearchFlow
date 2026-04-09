#!/usr/bin/env python3
"""Create a cross-platform local link under linkedCodebases/.

Usage examples:

- macOS / Linux:
  python3 scripts/link_codebase.py /path/to/your-codebase
  python3 scripts/link_codebase.py /path/to/your-codebase --name my-project

- Windows:
  py -3 scripts\link_codebase.py C:\path\to\your-codebase
  py -3 scripts\link_codebase.py C:\path\to\your-codebase --name my-project
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


def linked_codebases_dir() -> Path:
    return repo_root() / "linkedCodebases"


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


def remove_existing(target: Path, source: Path, force: bool) -> None:
    if not path_exists(target):
        return
    if points_to(target, source):
        return
    if not force:
        raise RuntimeError(
            f"{target} already exists and points elsewhere. "
            "Use --force to replace it."
        )
    if target.is_symlink() or target.is_file():
        target.unlink()
        return
    if target.is_dir():
        if is_windows() and is_reparse_point(target):
            target.rmdir()
            return
        if any(target.iterdir()):
            raise RuntimeError(
                f"{target} is a real non-empty directory and will not be removed automatically."
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Link an external code repository into linkedCodebases/."
    )
    parser.add_argument("source", help="path to the external code repository")
    parser.add_argument(
        "--name",
        help="alias name under linkedCodebases/ (defaults to the source folder name)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing link with the same name if it points elsewhere",
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        print(f"error: source path does not exist: {source}", file=sys.stderr)
        return 1
    if not source.is_dir():
        print(f"error: source path is not a directory: {source}", file=sys.stderr)
        return 1

    alias = args.name or source.name
    target = linked_codebases_dir() / alias

    try:
        remove_existing(target, source, force=args.force)
        create_dir_alias(source, target)
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{target.relative_to(repo_root())} -> {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
