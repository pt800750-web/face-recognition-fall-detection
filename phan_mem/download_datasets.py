from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
from urllib.parse import quote

import requests

from src.utils import (
    build_arg_parser,
    download_with_resume,
    ensure_project_dirs,
    extract_zip,
    load_config,
    md5_file,
    resolve_path,
    setup_logging,
)


def zenodo_file_info(record_id: str, archive_name: str) -> Dict[str, object]:
    api_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    files = payload.get("files", [])
    if not files:
        raise RuntimeError(f"No files found in Zenodo record {record_id}")

    selected: Optional[Dict[str, object]] = None
    for file_info in files:
        key = str(file_info.get("key", ""))
        if key.endswith(".zip") and (archive_name in key or archive_name.endswith(Path(key).name)):
            selected = file_info
            break
    if selected is None:
        for file_info in files:
            key = str(file_info.get("key", ""))
            if key.endswith(".zip"):
                selected = file_info
                break
    if selected is None:
        raise RuntimeError(f"No ZIP archive found in Zenodo record {record_id}")

    key = str(selected.get("key", ""))
    links = selected.get("links", {}) if isinstance(selected.get("links", {}), dict) else {}
    download_url = links.get("self") or links.get("download")
    if not download_url:
        download_url = f"https://zenodo.org/records/{record_id}/files/{quote(key, safe='')}?download=1"

    checksum = str(selected.get("checksum", ""))
    if checksum.startswith("md5:"):
        checksum = checksum.split(":", 1)[1]

    return {
        "key": key,
        "download_url": download_url,
        "size": int(selected.get("size", 0) or 0),
        "md5": checksum,
    }


def write_manual_note(dataset_name: str, root_dir: Path, content: str) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    note_path = root_dir / "MANUAL_DOWNLOAD.md"
    note_path.write_text(content.strip() + "\n", encoding="utf-8")


def download_gmdcsa24(config: Dict[str, object], force: bool, no_extract: bool, logger) -> None:
    dataset_cfg = config["datasets"]["gmdcsa24"]  # type: ignore[index]
    root_dir = resolve_path(dataset_cfg["root_dir"])
    downloads_dir = resolve_path(config["paths"]["raw_dir"]) / "downloads"  # type: ignore[index]
    downloads_dir.mkdir(parents=True, exist_ok=True)
    archive_path = downloads_dir / str(dataset_cfg["archive_name"])

    info = zenodo_file_info(str(dataset_cfg["zenodo_record_id"]), str(dataset_cfg["archive_name"]))
    expected_size = int(info["size"]) if int(info["size"]) > 0 else None
    expected_md5 = str(info["md5"] or dataset_cfg.get("expected_md5", "")).strip()

    if force and archive_path.exists():
        archive_path.unlink()
        logger.info("Removed existing archive because --force was used: %s", archive_path)

    download_with_resume(str(info["download_url"]), archive_path, expected_size, logger)
    if expected_md5:
        actual_md5 = md5_file(archive_path)
        if actual_md5.lower() != expected_md5.lower():
            raise RuntimeError(f"MD5 mismatch for {archive_path}: expected {expected_md5}, got {actual_md5}")
        logger.info("GMDCSA24 MD5 verified: %s", actual_md5)
    if not no_extract and bool(dataset_cfg.get("extract_after_download", True)):
        extract_zip(archive_path, root_dir, logger)
        logger.info("GMDCSA24 ready under %s", root_dir)


def download_up_fall_preliminary(config: Dict[str, object], force: bool, no_extract: bool, logger) -> None:
    dataset_cfg = config["datasets"]["up_fall"]  # type: ignore[index]
    if not bool(dataset_cfg.get("download_preliminary_sensor_zip", False)):
        return
    file_id = str(dataset_cfg["preliminary_drive_file_id"])
    root_dir = resolve_path(dataset_cfg["root_dir"])
    downloads_dir = resolve_path(config["paths"]["raw_dir"]) / "downloads"  # type: ignore[index]
    downloads_dir.mkdir(parents=True, exist_ok=True)
    archive_path = downloads_dir / "up_fall_preliminary_sensor.zip"
    if archive_path.exists() and not force:
        logger.info("UP-Fall preliminary archive already exists: %s", archive_path)
    else:
        try:
            import gdown
        except ImportError as exc:
            raise ImportError("gdown is required for Google Drive downloads. Install requirements.txt first.") from exc
        url = f"https://drive.google.com/uc?id={file_id}"
        logger.info("Downloading UP-Fall preliminary sensor zip with gdown: %s", file_id)
        gdown.download(url, str(archive_path), quiet=False, fuzzy=True)
    if not no_extract:
        extract_zip(archive_path, root_dir / "preliminary_sensor", logger)


def main() -> None:
    parser = build_arg_parser("Download or bootstrap all configured fall detection datasets.")
    parser.add_argument("--dataset", default="all", choices=["all", "gmdcsa24", "le2i", "up_fall"], help="Dataset to download/bootstrap.")
    parser.add_argument("--force", action="store_true", help="Re-download archives when supported.")
    parser.add_argument("--no-extract", action="store_true", help="Only download archives, do not extract.")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_project_dirs(config)
    logger = setup_logging(config, "download_datasets")

    if args.dataset in {"all", "gmdcsa24"} and bool(config["datasets"]["gmdcsa24"].get("enabled", True)):  # type: ignore[index]
        download_gmdcsa24(config, args.force, args.no_extract, logger)

    if args.dataset in {"all", "le2i"} and bool(config["datasets"]["le2i"].get("enabled", True)):  # type: ignore[index]
        le2i_cfg = config["datasets"]["le2i"]  # type: ignore[index]
        le2i_root = resolve_path(le2i_cfg["root_dir"])
        write_manual_note(
            "le2i",
            le2i_root,
            f"""
# LE2I / ImViA Manual Dataset Placement

Official page: {le2i_cfg.get("official_url")}

The official page currently describes the LE2I fall detection dataset but does not expose a stable direct download link.
Place extracted LE2I videos and annotation files anywhere under this directory:

```text
{le2i_root}
```

The reader accepts common structures such as `Home`, `Coffee room`, `Office`, `Lecture room`, `Fall`, `ADL`, and `Annotation_files`.
It pairs annotations by video stem and also supports pseudo labels from folder/file names when no annotation is available.
""",
        )
        logger.warning("LE2I requires manual placement. Wrote instructions to %s", le2i_root / "MANUAL_DOWNLOAD.md")

    if args.dataset in {"all", "up_fall"} and bool(config["datasets"]["up_fall"].get("enabled", True)):  # type: ignore[index]
        up_cfg = config["datasets"]["up_fall"]  # type: ignore[index]
        up_root = resolve_path(up_cfg["root_dir"])
        write_manual_note(
            "up_fall",
            up_root,
            f"""
# UP-Fall Manual Dataset Placement

Official page: {up_cfg.get("official_url")}

For vision training, download Camera1/Camera2 files from the HAR-UP page and place extracted image folders, camera zip files, or prebuilt videos under:

```text
{up_root}
```

The prepare pipeline can:
- scan existing videos,
- extract Camera zip files whose names contain `Camera` or `Cam`,
- convert image sequences into MP4 files under `data/cache/generated_videos/up_fall`,
- infer labels from Activity IDs A1-A11.

Activity mapping:
A1-A5 -> FALLING/FALLEN pseudo segments
A6-A7 -> WALKING_STANDING
A8 -> SITTING
A9 -> BENDING
A10 -> NORMAL
A11 -> LYING_INTENTIONAL

The preliminary sensor-only Google Drive zip can be downloaded by setting `datasets.up_fall.download_preliminary_sensor_zip: true`, but it is not enough for the camera-only model unless Camera files are also present.
""",
        )
        download_up_fall_preliminary(config, args.force, args.no_extract, logger)
        logger.warning("UP-Fall camera data is manual. Wrote instructions to %s", up_root / "MANUAL_DOWNLOAD.md")

    logger.info("Dataset bootstrap complete.")


if __name__ == "__main__":
    main()

