"""Command line interface.

    python -m src train
    python -m src train --epochs 300 --patch-size 96 --base-channels 32
    python -m src predict --checkpoint runs/unet/best.pt --image-dir data/fafb_em
    python -m src masks --output-dir data/fafb_em_masks
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Train a U-Net to segment circular structures in EM images.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ train
    train_parser = subparsers.add_parser(
        "train", help="train the segmentation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_group = train_parser.add_argument_group("data")
    data_group.add_argument("--image-dir", type=Path, default=Path("data/fafb_em"))
    data_group.add_argument("--label-dir", type=Path, default=Path("data/fafb_em_gt"))
    data_group.add_argument("--output-dir", type=Path, default=Path("runs/unet"))
    data_group.add_argument("--val-fraction", type=float, default=0.2)
    data_group.add_argument("--test-fraction", type=float, default=0.2)
    data_group.add_argument("--supersample", type=int, default=4,
                            help="subgrid used when rasterising circles to masks")
    data_group.add_argument("--patch-size", type=int, default=128)
    data_group.add_argument("--samples-per-epoch", type=int, default=200,
                            help="random crops drawn per epoch")
    data_group.add_argument("--no-augment", action="store_true")
    data_group.add_argument("--num-workers", type=int, default=0)

    model_group = train_parser.add_argument_group("model")
    model_group.add_argument("--base-channels", type=int, default=16)
    model_group.add_argument("--depth", type=int, default=3,
                             help="number of downsampling steps")
    model_group.add_argument("--dropout", type=float, default=0.0)

    optim_group = train_parser.add_argument_group("optimisation")
    optim_group.add_argument("--epochs", type=int, default=200)
    optim_group.add_argument("--batch-size", type=int, default=8)
    optim_group.add_argument("--lr", type=float, default=1e-3)
    optim_group.add_argument("--weight-decay", type=float, default=1e-4)
    optim_group.add_argument("--dice-weight", type=float, default=0.5,
                             help="0 = pure BCE, 1 = pure dice")
    optim_group.add_argument("--pos-weight", type=float, default=None,
                             help="BCE positive-class weight (default: from class balance)")
    optim_group.add_argument("--threshold", type=float, default=0.5)
    optim_group.add_argument("--patience", type=int, default=50,
                             help="early stopping patience in epochs (0 disables)")

    misc_group = train_parser.add_argument_group("misc")
    misc_group.add_argument("--seed", type=int, default=42)
    misc_group.add_argument("--run-id", default=None,
                            help="tag for this run's checkpoints (default: random, "
                                 "e.g. yhtkfne-best.pt / yhtkfne-last.pt)")
    misc_group.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    misc_group.add_argument("--log-every", type=int, default=10)
    misc_group.add_argument("--no-overlays", action="store_true")

    # ---------------------------------------------------------------- predict
    predict_parser = subparsers.add_parser(
        "predict", help="segment a folder of images with a trained checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    predict_parser.add_argument("--checkpoint", type=Path, default=None,
                                help="path to a *-best.pt / *-last.pt "
                                     "(default: newest *-best.pt in --run-dir)")
    predict_parser.add_argument("--run-dir", type=Path, default=Path("runs/unet"),
                                help="where to look when --checkpoint is not given")
    predict_parser.add_argument("--image-dir", type=Path, default=Path("data/fafb_em"))
    predict_parser.add_argument("--output-dir", type=Path, default=Path("runs/unet/predictions"))
    predict_parser.add_argument("--threshold", type=float, default=None,
                                help="default: the threshold stored in the checkpoint")
    predict_parser.add_argument("--device", default="auto")

    # ------------------------------------------------------------------ masks
    masks_parser = subparsers.add_parser(
        "masks", help="rasterise the annotation csvs to mask PNGs (sanity check)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    masks_parser.add_argument("--image-dir", type=Path, default=Path("data/fafb_em"))
    masks_parser.add_argument("--label-dir", type=Path, default=Path("data/fafb_em_gt"))
    masks_parser.add_argument("--output-dir", type=Path, default=Path("data/fafb_em_masks"))
    masks_parser.add_argument("--supersample", type=int, default=4)

    return parser


def export_masks(args):
    """Write the rasterised masks out so the labels can be eyeballed."""
    from .data import load_samples
    from .visualise import save_mask, save_prediction_overlay

    samples = load_samples(args.image_dir, args.label_dir, supersample=args.supersample)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        save_mask(args.output_dir / f"{sample.name}_mask.png", sample.mask)
        save_prediction_overlay(
            args.output_dir / f"{sample.name}_mask_overlay.png", sample.image, sample.mask
        )
    print(f"wrote {len(samples)} mask(s) to {args.output_dir}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        from .train import train
        train(args)
    elif args.command == "predict":
        from .predict import predict
        predict(args)
    elif args.command == "masks":
        export_masks(args)
    return 0
