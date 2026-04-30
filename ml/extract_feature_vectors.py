from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import datasets


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.feature_layer import FEATURE_NAMES, extract_feature_vector_bundle


DEFAULT_DATASET_ROOT = BASE_DIR / "data" / "image_dataset"
DEFAULT_OUTPUT_DIR = BASE_DIR / "artifacts" / "feature_vectors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract raw and normalized feature vectors for the A-Eye dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory that contains train/ and test/ image folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where feature CSVs and normalization stats will be written.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of the train folder reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the reproducible train/validation split.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional cap on extracted train samples for smoke testing.",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=None,
        help="Optional cap on extracted validation samples for smoke testing.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Optional cap on extracted test samples for smoke testing.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.val_split < 1.0:
        raise ValueError("--val-split must be between 0 and 1.")


def maybe_limit_indices(indices: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    limited = rng.choice(indices, size=limit, replace=False)
    return np.sort(limited)

def build_split_samples(
    dataset_root: Path,
    val_split: float,
    seed: int,
    train_limit: int | None,
    val_limit: int | None,
    test_limit: int | None,
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, int]]:
    train_source = datasets.ImageFolder(dataset_root / "train")
    test_source = datasets.ImageFolder(dataset_root / "test")
    targets = np.array(train_source.targets)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_split,
        random_state=seed,
    )
    train_indices, val_indices = next(splitter.split(np.zeros(len(targets)), targets))

    train_indices = maybe_limit_indices(train_indices, train_limit, seed)
    val_indices = maybe_limit_indices(val_indices, val_limit, seed + 1)
    test_indices = maybe_limit_indices(np.arange(len(test_source)), test_limit, seed + 2)

    splits = {
        "train": [train_source.samples[index] for index in train_indices],
        "val": [train_source.samples[index] for index in val_indices],
        "test": [test_source.samples[index] for index in test_indices],
    }
    return splits, train_source.class_to_idx


def extract_rows(
    samples: list[tuple[str, int]],
    split_name: str,
    dataset_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_path, label in samples:
        with Image.open(sample_path) as image:
            bundle = extract_feature_vector_bundle(image)
        row = {
            "split": split_name,
            "label": label,
            "path": str(Path(sample_path).relative_to(dataset_root)),
            **bundle.raw.as_dict(),
        }
        rows.append(row)
    return rows

def fit_standardization_profile(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    matrix = np.asarray(
        [[float(row[feature_name]) for feature_name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)

    return {
        feature_name: {
            "mean": float(means[index]),
            "std": float(stds[index]),
        }
        for index, feature_name in enumerate(FEATURE_NAMES)
    }

def apply_standardization(
    rows: list[dict[str, Any]],
    profile: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_row = {
            "split": row["split"],
            "label": row["label"],
            "path": row["path"],
        }
        for feature_name in FEATURE_NAMES:
            mean = profile[feature_name]["mean"]
            std = profile[feature_name]["std"]
            normalized_row[feature_name] = float((float(row[feature_name]) - mean) / std)
        normalized_rows.append(normalized_row)
    return normalized_rows

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = ["split", "label", "path", *FEATURE_NAMES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    args = parse_args()
    validate_args(args)

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits, class_to_idx = build_split_samples(
        dataset_root=dataset_root,
        val_split=args.val_split,
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
    )

    raw_rows_by_split = {
        split_name: extract_rows(samples, split_name, dataset_root)
        for split_name, samples in splits.items()
    }
    train_profile = fit_standardization_profile(raw_rows_by_split["train"])
    normalized_rows_by_split = {
        split_name: apply_standardization(rows, train_profile)
        for split_name, rows in raw_rows_by_split.items()
    }

    for split_name in ("train", "val", "test"):
        write_csv(output_dir / f"{split_name}_raw.csv", raw_rows_by_split[split_name])
        write_csv(
            output_dir / f"{split_name}_normalized.csv",
            normalized_rows_by_split[split_name],
        )

    manifest = {
        "dataset_root": str(dataset_root),
        "class_to_idx": class_to_idx,
        "feature_names": list(FEATURE_NAMES),
        "split_sizes": {
            split_name: len(rows) for split_name, rows in raw_rows_by_split.items()
        },
        "normalization_profile": train_profile,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("Feature extraction complete.")
    print(f"Output directory: {output_dir}")
    print(f"Split sizes: {manifest['split_sizes']}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
if __name__ == "__main__":
    main()
