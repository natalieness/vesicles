"""Run a trained checkpoint over a folder of images."""

from __future__ import annotations

from pathlib import Path

import torch

from .data import IMAGE_EXTENSIONS, load_image
from .model import build_model
from .visualise import save_mask, save_prediction_overlay


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def find_latest_checkpoint(run_dir, pattern="*-best.pt"):
    """Most recently written checkpoint under `run_dir`.

    Searches recursively, because each run writes into its own {run_id} folder.
    """
    run_dir = Path(run_dir)
    candidates = sorted(run_dir.rglob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"no {pattern} found in {run_dir} - train first, or pass --checkpoint"
        )
    if len(candidates) > 1:
        print(f"{len(candidates)} checkpoints under {run_dir}, using the newest")
    return candidates[-1]


def predict(args):
    from .train import get_device, predict_mask

    device = get_device(args.device)
    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.run_dir)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    stats = checkpoint["statistics"]
    threshold = args.threshold if args.threshold is not None else checkpoint["threshold"]
    print(f"loaded {checkpoint_path} (run {checkpoint.get('run_id', '?')}, "
          f"epoch {checkpoint['epoch']}, threshold {threshold})")

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"no images found in {image_dir}")

    for path in images:
        image = load_image(path)
        mask = predict_mask(model, image, stats, device, threshold, model.size_multiple)
        save_mask(output_dir / f"{path.stem}_mask.png", mask)
        save_prediction_overlay(output_dir / f"{path.stem}_pred.png", image, mask)
        print(f"  {path.name}: {mask.mean():.2%} foreground")

    print(f"wrote {len(images)} prediction(s) to {output_dir}")
