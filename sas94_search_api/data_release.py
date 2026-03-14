from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib import request

try:
    import certifi
except Exception:
    certifi = None


DEFAULT_REPO = "kor-noah-han/sas-94-search-API"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract SAS search runtime data from GitHub Releases.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository in OWNER/NAME form.")
    parser.add_argument("--tag", help="Release tag to download. Defaults to the latest release.")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where the release archive should be extracted.",
    )
    parser.add_argument(
        "--archive-name",
        help="Explicit archive filename. Defaults to the .tar.gz asset attached to the release.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing extracted files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the release and print asset URLs without downloading.",
    )
    return parser.parse_args()


def get_json(url: str) -> dict[str, object]:
    req = request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "sas94-search-api"})
    with request.urlopen(req, timeout=60, context=ssl_context()) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_release(repo: str, tag: str | None) -> dict[str, object]:
    if tag:
        url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    else:
        url = f"https://api.github.com/repos/{repo}/releases/latest"
    return get_json(url)


def pick_asset(release: dict[str, object], suffix: str, explicit_name: str | None = None) -> dict[str, object]:
    assets = release.get("assets", [])
    if not isinstance(assets, list):
        raise RuntimeError("Release assets payload is invalid.")
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name", ""))
        if explicit_name and name == explicit_name:
            return asset
        if not explicit_name and name.endswith(suffix):
            return asset
    raise RuntimeError(f"Could not find release asset matching {explicit_name or suffix}.")


def download_file(url: str, destination: Path) -> None:
    req = request.Request(url, headers={"User-Agent": "sas94-search-api"})
    with request.urlopen(req, timeout=300, context=ssl_context()) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def ssl_context() -> ssl.SSLContext | None:
    if certifi is None:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_sha256(checksum_path: Path, archive_name: str) -> str:
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[-1].endswith(archive_name):
            return parts[0]
    raise RuntimeError("Could not parse checksum file.")


def is_within_directory(directory: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def extract_archive(archive_path: Path, output_dir: Path, *, force: bool) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            member_path = Path(member.name)
            relative_parts = member_path.parts[1:]
            if not relative_parts:
                continue
            target_path = output_dir.joinpath(*relative_parts)
            if not is_within_directory(output_dir, target_path):
                raise RuntimeError(f"Refusing to extract outside output directory: {member.name}")
            if target_path.exists() and not force:
                raise RuntimeError(f"Refusing to overwrite existing file: {target_path}")

        for member in members:
            member_path = Path(member.name)
            relative_parts = member_path.parts[1:]
            if not relative_parts:
                continue
            member.name = str(Path(*relative_parts))
            tar.extract(member, path=output_dir)


def main() -> int:
    args = parse_args()
    release = resolve_release(args.repo, args.tag)
    archive_asset = pick_asset(release, ".tar.gz", args.archive_name)
    checksum_asset = pick_asset(release, ".sha256")
    tag_name = str(release.get("tag_name", ""))

    archive_url = str(archive_asset.get("browser_download_url", ""))
    checksum_url = str(checksum_asset.get("browser_download_url", ""))
    if not archive_url or not checksum_url:
        raise RuntimeError("Release assets are missing download URLs.")

    payload = {
        "repo": args.repo,
        "tag": tag_name,
        "archive": archive_asset.get("name"),
        "archive_url": archive_url,
        "checksum_url": checksum_url,
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sas94-search-data-") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        archive_path = tmp_dir / str(archive_asset["name"])
        checksum_path = tmp_dir / str(checksum_asset["name"])

        download_file(archive_url, archive_path)
        download_file(checksum_url, checksum_path)

        expected = expected_sha256(checksum_path, archive_path.name)
        actual = sha256_file(archive_path)
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for {archive_path.name}: expected {expected}, got {actual}"
            )

        extract_archive(archive_path, output_dir, force=args.force)

    print(
        json.dumps(
            {
                "repo": args.repo,
                "tag": tag_name,
                "output_dir": str(output_dir),
                "archive": archive_asset.get("name"),
                "sha256": actual,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
