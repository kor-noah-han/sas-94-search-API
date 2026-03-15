#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


INCLUDE_PATHS = (
    "data/config/sas-ko-en-terms.json",
    "data/processed/sas-rag/search/sas9-pdf-fts.db",
    "data/processed/sas-rag/search/sas9-pdf-route-index.json",
    "data/processed/sas-rag/corpus/sas9-pdf-corpus.jsonl",
    "data/qdrant/sas9_pdf/meta.json",
    "data/qdrant/sas9_pdf/collection/sas9_pdf_chunks/storage.sqlite",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a release bundle for SAS search runtime data.")
    parser.add_argument(
        "--source-root",
        required=True,
        help="Project root that contains the built search data files.",
    )
    parser.add_argument(
        "--output-dir",
        default="dist",
        help="Directory where the release bundle should be written.",
    )
    parser.add_argument(
        "--version",
        default=datetime.now(timezone.utc).strftime("%Y%m%d"),
        help="Bundle version suffix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_stem = f"sas94-search-data-{args.version}"
    staging_root = Path(archive_stem)
    archive_path = output_dir / f"{archive_stem}.tar.gz"
    checksum_path = output_dir / f"{archive_stem}.sha256"

    files: list[tuple[Path, Path]] = []
    manifest_files: list[dict[str, object]] = []
    total_bytes = 0

    for relative_str in INCLUDE_PATHS:
        relative_path = Path(relative_str)
        source_path = source_root / relative_path
        if not source_path.exists():
            raise SystemExit(f"Missing required file: {source_path}")
        archive_member = staging_root / relative_path
        file_size = source_path.stat().st_size
        total_bytes += file_size
        files.append((source_path, archive_member))
        manifest_files.append(
            {
                "path": relative_path.as_posix(),
                "bytes": file_size,
                "sha256": sha256_file(source_path),
            }
        )

    manifest = {
        "bundle": archive_stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "total_bytes": total_bytes,
        "files": manifest_files,
    }

    with tarfile.open(archive_path, "w:gz") as tar:
        for source_path, archive_member in files:
            tar.add(source_path, arcname=archive_member.as_posix(), recursive=False)
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        manifest_info = tarfile.TarInfo(name=(staging_root / "manifest.json").as_posix())
        manifest_info.size = len(manifest_bytes)
        manifest_info.mtime = int(datetime.now(timezone.utc).timestamp())
        tar.addfile(manifest_info, fileobj=io.BytesIO(manifest_bytes))

    archive_sha = sha256_file(archive_path)
    checksum_path.write_text(f"{archive_sha}  {archive_path.name}\n", encoding="utf-8")

    print(json.dumps(
        {
            "archive": str(archive_path),
            "checksum_file": str(checksum_path),
            "total_bytes": total_bytes,
            "archive_sha256": archive_sha,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
