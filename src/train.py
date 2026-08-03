"""Training / evaluation loop for the vesicle segmentation U-Net."""

from __future__ import annotations

import csv
import json
import random
import secrets
import string
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import (
    FullImageDataset,
    PatchDataset,
    dataset_statistics,
    load_samples,
    split_samples,
)
from .model import build_model
from .visualise import save_prediction_overlay


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_run_id(length=7):
    """Random tag for a run's checkpoints, e.g. 'yhtkfne'.

    Deliberately drawn from OS entropy rather than the seeded global RNG, so two
    runs with the same --seed still get distinct filenames.
    """
    return "".join(secrets.choice(string.ascii_lowercase) for _ in range(length))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(logits, targets, eps=1.0):
    """Soft dice over the whole batch; complements BCE under class imbalance."""
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    total = probs.sum() + targets.sum()
    return 1.0 - (2.0 * intersection + eps) / (total + eps)


class CombinedLoss(nn.Module):
    """BCE (pixel-wise, imbalance-aware) + soft dice (region overlap)."""

    def __init__(self, pos_weight=None, dice_weight=0.5):
        super().__init__()
        weight = None if pos_weight is None else torch.tensor(float(pos_weight))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight)
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        loss = (1.0 - self.dice_weight) * self.bce(logits, targets)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * dice_loss(logits, targets)
        return loss


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, threshold=0.5):
    """Aggregate TP/FP/FN over the split, then derive metrics once."""
    model.eval()
    tp = fp = fn = 0.0
    total_loss = 0.0
    n_batches = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total_loss += float(loss_fn(logits, masks))
        n_batches += 1

        preds = (torch.sigmoid(logits) >= threshold).float()
        tp += float((preds * masks).sum())
        fp += float((preds * (1 - masks)).sum())
        fn += float(((1 - preds) * masks).sum())

    eps = 1e-8
    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
    }


def format_metrics(metrics):
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def train(args):
    seed_everything(args.seed)
    device = get_device(args.device)

    # Everything this run produces lives in its own {run_id} directory, so runs
    # never overwrite each other's checkpoints, metrics or predictions.
    run_id = args.run_id or generate_run_id()
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / f"{run_id}-best.pt"
    last_path = run_dir / f"{run_id}-last.pt"

    print(f"run id: {run_id}  (outputs in {run_dir})")
    print(f"device: {device}")
    print(f"loading data from {args.image_dir} + {args.label_dir}")
    samples = load_samples(args.image_dir, args.label_dir, supersample=args.supersample)
    splits = split_samples(
        samples,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    for name, split in splits.items():
        print(f"{name}: {len(split)} image(s) -> {[s.name for s in split]}")
    if len(splits["train"]) < 2:
        print(
            "WARNING: fewer than 2 training images. Metrics will be extremely noisy "
            "and the model will likely memorise the training image."
        )

    stats = dataset_statistics(splits["train"])
    print(
        f"train stats: mean={stats['mean']:.4f} std={stats['std']:.4f} "
        f"foreground={stats['foreground_fraction']:.2%} pos_weight={stats['pos_weight']:.2f}"
    )

    model = build_model(
        base_channels=args.base_channels, depth=args.depth, dropout=args.dropout
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: U-Net depth={args.depth} base_channels={args.base_channels} "
          f"({n_params:,} parameters)")

    train_dataset = PatchDataset(
        splits["train"],
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        mean=stats["mean"],
        std=stats["std"],
        augment=not args.no_augment,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    eval_loaders = {
        name: DataLoader(
            FullImageDataset(
                split, mean=stats["mean"], std=stats["std"], multiple=model.size_multiple
            ),
            batch_size=1,
            shuffle=False,
        )
        for name, split in splits.items()
        if split
    }

    pos_weight = stats["pos_weight"] if args.pos_weight is None else args.pos_weight
    loss_fn = CombinedLoss(
        pos_weight=None if pos_weight <= 0 else pos_weight, dice_weight=args.dice_weight
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(args.patience // 3, 2)
    )

    history_path = run_dir / f"{run_id}-history.csv"
    history_fields = [
        "epoch", "train_loss", "val_loss", "val_dice", "val_iou",
        "val_precision", "val_recall", "lr",
    ]
    history_file = open(history_path, "w", newline="")
    history_writer = csv.DictWriter(history_file, fieldnames=history_fields)
    history_writer.writeheader()

    best_dice = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        n_batches = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), masks)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)

        val_metrics = evaluate(
            model, eval_loaders["val"], loss_fn, device, threshold=args.threshold
        )
        scheduler.step(val_metrics["dice"])

        history_writer.writerow({
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_metrics['loss']:.6f}",
            "val_dice": f"{val_metrics['dice']:.6f}",
            "val_iou": f"{val_metrics['iou']:.6f}",
            "val_precision": f"{val_metrics['precision']:.6f}",
            "val_recall": f"{val_metrics['recall']:.6f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })
        history_file.flush()

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"epoch {epoch:4d}/{args.epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  val_dice={val_metrics['dice']:.4f}  "
                f"val_iou={val_metrics['iou']:.4f}"
            )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, args, stats, epoch, val_metrics, run_id)
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"early stopping at epoch {epoch} (no val improvement "
                      f"for {args.patience} epochs)")
                break

    history_file.close()
    save_checkpoint(last_path, model, args, stats, epoch, val_metrics, run_id)
    print(f"training finished in {time.time() - start:.1f}s; "
          f"best val dice {best_dice:.4f} at epoch {best_epoch}")

    # Final evaluation with the best weights, on every split.
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    results = {}
    for name, loader in eval_loaders.items():
        results[name] = evaluate(model, loader, loss_fn, device, threshold=args.threshold)
        print(f"{name:>5}: {format_metrics(results[name])}")

    summary = {
        "run_id": run_id,
        "checkpoints": {"best": best_path.name, "last": last_path.name},
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "threshold": args.threshold,
        "splits": {name: [s.name for s in split] for name, split in splits.items()},
        "train_statistics": stats,
        "metrics": results,
        "config": vars(args).copy(),
    }
    summary["config"] = {k: str(v) if isinstance(v, Path) else v
                         for k, v in summary["config"].items()}
    results_path = run_dir / f"{run_id}-results.json"
    with open(results_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"wrote {results_path.name}")

    if not args.no_overlays:
        # Per-run folder and filenames, so repeated runs in one output dir keep
        # their predictions separate and traceable to a checkpoint.
        overlay_dir = run_dir / "test_predictions"
        overlay_dir.mkdir(exist_ok=True)
        for sample in splits["test"]:
            save_prediction_overlay(
                overlay_dir / f"{sample.name}_{best_path.stem}_pred.png",
                sample.image,
                predict_mask(model, sample.image, stats, device, args.threshold,
                             model.size_multiple),
                truth=sample.mask,
            )
        print(f"wrote test overlays to {overlay_dir}")

    print(f"outputs in {run_dir}")
    return summary


@torch.no_grad()
def predict_mask(model, image, stats, device, threshold, multiple):
    """Binary mask for one whole image, cropped back to the original size."""
    from .data import pad_to_multiple

    model.eval()
    normalised = (image - stats["mean"]) / stats["std"]
    tensor = torch.from_numpy(normalised.astype(np.float32)).unsqueeze(0)
    tensor, (h, w) = pad_to_multiple(tensor, multiple)
    logits = model(tensor.unsqueeze(0).to(device))
    probs = torch.sigmoid(logits)[0, 0, :h, :w].cpu().numpy()
    return (probs >= threshold).astype(np.uint8)


def save_checkpoint(path, model, args, stats, epoch, metrics, run_id=None):
    torch.save(
        {
            "run_id": run_id,
            "model_state": model.state_dict(),
            "statistics": stats,
            "epoch": epoch,
            "metrics": metrics,
            "model_config": {
                "base_channels": args.base_channels,
                "depth": args.depth,
                "dropout": args.dropout,
            },
            "threshold": args.threshold,
        },
        path,
    )
