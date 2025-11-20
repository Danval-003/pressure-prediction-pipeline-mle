"""Utility to expose local package sources without installing them.

Importing this module inserts every ``packages/*/src`` directory into
``sys.path`` so that ``import crispdm_...`` works while developing
inside the repository.
"""

from __future__ import annotations

from pathlib import Path
import sys


def bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent
    packages_dir = repo_root / "packages"
    if not packages_dir.exists():
        return

    for pkg_dir in packages_dir.iterdir():
        src_path = pkg_dir / "src"
        if not src_path.exists():
            continue

        path_str = str(src_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


bootstrap()
