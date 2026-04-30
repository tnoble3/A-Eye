from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.cnn_baseline import AEyeBaselineCNN


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = BASE_DIR / "data" / "image_dataset"
DEFAULT_OUTPUT_DIR = BASE_DIR / "artifacts" / "cnn_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the A-Eye baseline CNN on the image dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory that contains train/ and test/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where checkpoints and metrics will be written.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Image size passed to the baseline CNN.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of the train folder reserved for validation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for data splitting and training.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Training device. 'auto' prefers CUDA, then CPU. Use 'mps' explicitly to try Apple Metal.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional cap on train samples for quick smoke tests.",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=None,
        help="Optional cap on validation samples for quick smoke tests.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Optional cap on test samples for quick smoke tests.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.image_size < 8:
        raise ValueError("--image-size must be at least 8.")
    if not 0.0 < args.val_split < 1.0:
        raise ValueError("--val-split must be between 0 and 1.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(device_name)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    common_steps = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    return transforms.Compose(common_steps), transforms.Compose(common_steps)


def maybe_limit_indices(indices: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    limited = rng.choice(indices, size=limit, replace=False)
    return np.sort(limited)


def build_datasets(
    dataset_root: Path,
    image_size: int,
    val_split: float,
    seed: int,
    train_limit: int | None,
    val_limit: int | None,
    test_limit: int | None,
) -> tuple[Subset, Subset, Subset, dict[str, int]]:
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_transform, eval_transform = build_transforms(image_size)
    train_source = datasets.ImageFolder(train_dir, transform=train_transform)
    eval_source = datasets.ImageFolder(train_dir, transform=eval_transform)
    test_source = datasets.ImageFolder(test_dir, transform=eval_transform)

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

    train_dataset = Subset(train_source, train_indices.tolist())
    val_dataset = Subset(eval_source, val_indices.tolist())
    test_dataset = Subset(test_source, test_indices.tolist())

    return train_dataset, val_dataset, test_dataset, train_source.class_to_idx


def build_loader(
    dataset: Subset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> int:
    predictions = (logits >= 0).long()
    return int((predictions == labels.long()).sum().item())


def validate_loss(loss_value: float, stage: str, device: torch.device) -> None:
    if not np.isfinite(loss_value):
        raise RuntimeError(
            f"Encountered a non-finite {stage} loss on device '{device}'. "
            "Retry with --device cpu or --device cuda."
        )
    if loss_value < -1e-6:
        raise RuntimeError(
            f"Encountered an invalid negative {stage} BCE loss on device '{device}'. "
            "Retry with --device cpu or --device cuda."
        )


def move_batch_to_device(
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return (
        images.to(device, non_blocking=non_blocking),
        labels.to(device=device, dtype=torch.float32, non_blocking=non_blocking),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images, labels = move_batch_to_device(images, labels, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_value = float(loss.item())
        validate_loss(loss_value, "train", device)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss_value * batch_size
        total_correct += accuracy_from_logits(logits, labels)
        total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images, labels = move_batch_to_device(images, labels, device)

        logits = model(images)
        loss = criterion(logits, labels)
        loss_value = float(loss.item())
        validate_loss(loss_value, "evaluation", device)

        batch_size = labels.size(0)
        total_loss += loss_value * batch_size
        total_correct += accuracy_from_logits(logits, labels)
        total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    config = vars(args).copy()
    config["dataset_root"] = str(config["dataset_root"])
    config["output_dir"] = str(config["output_dir"])
    return config


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    class_to_idx: dict[str, int],
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "class_to_idx": class_to_idx,
        "config": serializable_config(args),
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"

    train_dataset, val_dataset, test_dataset, class_to_idx = build_datasets(
        dataset_root=dataset_root,
        image_size=args.image_size,
        val_split=args.val_split,
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
    )

    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = build_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = AEyeBaselineCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Using device: {device}")
    print(f"Classes: {class_to_idx}")
    print(
        "Dataset sizes:",
        {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
    )

    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_val_accuracy = -1.0
    best_checkpoint_path = output_dir / "best_model.pt"
    last_checkpoint_path = output_dir / "last_model.pt"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_seconds = time.perf_counter() - epoch_start

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(epoch_summary)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"time={epoch_seconds:.1f}s"
        )

        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            epoch,
            epoch_summary,
            class_to_idx,
            args,
        )
        if val_metrics["accuracy"] >= best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch,
                epoch_summary,
                class_to_idx,
                args,
            )

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    summary = {
        "config": serializable_config(args),
        "device": str(device),
        "class_to_idx": class_to_idx,
        "dataset_sizes": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "history": history,
        "test": test_metrics,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        f"Training complete. Best epoch: {best_epoch}. "
        f"Test loss={test_metrics['loss']:.4f}, test_acc={test_metrics['accuracy']:.4f}"
    )
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Metrics summary: {metrics_path}")


if __name__ == "__main__":
    main()
