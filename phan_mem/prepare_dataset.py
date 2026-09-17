from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List

from src.dataset_readers import build_readers
from src.label_mapping import ID_TO_LABEL, UNIFIED_LABELS
from src.utils import (
    build_arg_parser,
    ensure_project_dirs,
    load_config,
    resolve_path,
    set_seed,
    setup_logging,
    write_json,
    write_jsonl,
)


def split_manifest(rows: List[Dict[str, object]], config: Dict[str, object]) -> Dict[str, List[Dict[str, object]]]:
    train_ratio = float(config["prepare"]["train_ratio"])  # type: ignore[index]
    val_ratio = float(config["prepare"]["val_ratio"])  # type: ignore[index]
    seed = int(config["project"]["seed"])  # type: ignore[index]
    rng = random.Random(seed)

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        dataset_name = str(row.get("dataset_name", "unknown"))
        subject_id = str(row.get("subject_id", "unknown_subject"))
        grouped[f"{dataset_name}:{subject_id}"].append(row)

    subject_keys = sorted(grouped)
    rng.shuffle(subject_keys)
    if len(subject_keys) >= 3:
        train_count = max(1, int(round(len(subject_keys) * train_ratio)))
        val_count = max(1, int(round(len(subject_keys) * val_ratio)))
        if train_count + val_count >= len(subject_keys):
            train_count = max(1, len(subject_keys) - 2)
            val_count = 1
        train_keys = set(subject_keys[:train_count])
        val_keys = set(subject_keys[train_count : train_count + val_count])
        test_keys = set(subject_keys[train_count + val_count :])
        return {
            "train": [row for key in train_keys for row in grouped[key]],
            "val": [row for key in val_keys for row in grouped[key]],
            "test": [row for key in test_keys for row in grouped[key]],
        }

    shuffled = rows[:]
    rng.shuffle(shuffled)
    train_end = max(1, int(round(len(shuffled) * train_ratio)))
    val_end = min(len(shuffled), train_end + max(1, int(round(len(shuffled) * val_ratio))))
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def label_stats(rows: List[Dict[str, object]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for segment in row.get("segments", []):
            if isinstance(segment, dict):
                counter[str(segment.get("label", "UNKNOWN"))] += 1
    return {label: int(counter.get(label, 0)) for label in UNIFIED_LABELS}


def main() -> None:
    parser = build_arg_parser("Prepare unified multi-dataset fall detection manifests.")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_project_dirs(config)
    set_seed(int(config["project"]["seed"]))
    logger = setup_logging(config, "prepare_dataset")

    readers = build_readers(config, logger)
    if not readers:
        raise RuntimeError("No dataset readers are enabled. Check configs/config.yaml -> datasets.enabled.")

    all_rows: List[Dict[str, object]] = []
    all_bad_files: List[Dict[str, str]] = []
    for reader in readers:
        logger.info("Scanning dataset: %s at %s", reader.dataset_name, reader.root_dir)
        result = reader.scan()
        logger.info("%s valid samples: %d | bad files: %d", reader.dataset_name, len(result.samples), len(result.bad_files))
        all_rows.extend(result.samples)
        all_bad_files.extend(result.bad_files)

    if not all_rows:
        manual_hints = {
            "gmdcsa24": config["datasets"]["gmdcsa24"].get("manual_url"),  # type: ignore[index]
            "le2i": config["datasets"]["le2i"].get("official_url"),  # type: ignore[index]
            "up_fall": config["datasets"]["up_fall"].get("official_url"),  # type: ignore[index]
        }
        write_json(resolve_path(config["paths"]["manifest_dir"]) / "bad_files.json", all_bad_files)  # type: ignore[index]
        raise RuntimeError(
            "No valid samples were found. Download/extract datasets under data/raw first. "
            f"Manual source hints: {manual_hints}"
        )

    all_rows.sort(key=lambda item: (str(item["dataset_name"]), str(item["subject_id"]), str(item["sample_id"])))
    splits = split_manifest(all_rows, config)
    manifest_dir = resolve_path(config["paths"]["manifest_dir"])  # type: ignore[index]
    write_jsonl(manifest_dir / "unified_all.jsonl", all_rows)
    for split_name, split_rows in splits.items():
        split_rows.sort(key=lambda item: (str(item["dataset_name"]), str(item["subject_id"]), str(item["sample_id"])))
        write_jsonl(manifest_dir / f"{split_name}.jsonl", split_rows)
        logger.info("%s samples: %d", split_name, len(split_rows))

    dataset_counts = Counter(str(row["dataset_name"]) for row in all_rows)
    stats = {
        "total_valid_samples": len(all_rows),
        "total_bad_files": len(all_bad_files),
        "datasets": {name: int(count) for name, count in sorted(dataset_counts.items())},
        "splits": {name: len(rows) for name, rows in splits.items()},
        "labels_all": label_stats(all_rows),
        "labels_by_split": {name: label_stats(rows) for name, rows in splits.items()},
        "unified_label_ids": {label: index for index, label in ID_TO_LABEL.items()},
    }
    write_json(manifest_dir / "dataset_stats.json", stats)
    write_json(manifest_dir / "bad_files.json", all_bad_files)
    logger.info("Unified manifest written to %s", manifest_dir / "unified_all.jsonl")
    logger.info("Dataset stats: %s", stats)
    logger.info("Bad file report written to %s", manifest_dir / "bad_files.json")


if __name__ == "__main__":
    main()

