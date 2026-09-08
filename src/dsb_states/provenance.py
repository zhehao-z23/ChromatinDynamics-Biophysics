from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path


def hash_records(paths: Iterable[str | Path], root: str | Path | None = None) -> list[dict]:
    root_path = Path(root).resolve() if root is not None else None
    records: list[dict] = []
    for value in sorted({Path(path).resolve() for path in paths}, key=str):
        digest = hashlib.sha256()
        with value.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        relative = None
        if root_path is not None:
            try:
                relative = str(value.relative_to(root_path)).replace("\\", "/")
            except ValueError:
                pass
        records.append(
            {
                "path": str(value),
                "relative_path": relative,
                "bytes": value.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return records


def write_hash_manifest(
    paths: Iterable[str | Path], output_path: str | Path, root: str | Path | None = None
) -> None:
    records = hash_records(paths, root=root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
