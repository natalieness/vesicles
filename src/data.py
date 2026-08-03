"""Images + circle annotations -> binary segmentation masks and torch datasets.

Labels are the csv files written by `generate_ground_truth.py` (one row per
annotated circle: index, image, x, y, radius). They are rasterised into a
binary object / no-object mask at load time, so the csvs stay the source of
truth and mask generation can be changed without re-annotating.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
LABEL_SUFFIXES = ("_anns.csv", ".csv")


# --------------------------------------------------------------------- loading


def find_pairs(image_dir, label_dir):
    """Match each image to its annotation csv. Returns (pairs, unlabelled)."""
    image_dir, label_dir = Path(image_dir), Path(label_dir)
    pairs, unlabelled = [], []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        for suffix in LABEL_SUFFIXES:
            label_path = label_dir / f"{image_path.stem}{suffix}"
            if label_path.exists():
                pairs.append((image_path, label_path))
                break
        else:
            unlabelled.append(image_path)
    return pairs, unlabelled


def load_circles(csv_path):
    """(N, 3) array of x, y, radius in pixel coordinates."""
    circles = []
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            circles.append((float(row["x"]), float(row["y"]), float(row["radius"])))
    return np.asarray(circles, dtype=np.float32).reshape(-1, 3)


def rasterize_circles(shape, circles, supersample=4, threshold=0.5):
    """Binary mask of `shape` (h, w) with the circles filled in.

    Radii here are only a few pixels, so each pixel's coverage is estimated on a
    `supersample`^2 subgrid and thresholded rather than testing pixel centres --
    that keeps small vesicles from losing whole rows to rounding.
    """
    height, width = shape
    coverage = np.zeros((height, width), dtype=np.float32)
    s = int(supersample)
    offsets = (np.arange(s, dtype=np.float32) + 0.5) / s - 0.5

    for x, y, r in circles:
        if r <= 0:
            continue
        # Work in a local window; annotator coords put pixel centres on integers.
        x0 = max(int(math.floor(x - r)), 0)
        x1 = min(int(math.ceil(x + r)) + 1, width)
        y0 = max(int(math.floor(y - r)), 0)
        y1 = min(int(math.ceil(y + r)) + 1, height)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = (np.arange(x0, x1, dtype=np.float32)[:, None] + offsets[None, :]).ravel()
        ys = (np.arange(y0, y1, dtype=np.float32)[:, None] + offsets[None, :]).ravel()
        inside = ((xs[None, :] - x) ** 2 + (ys[:, None] - y) ** 2) <= r * r
        local = inside.reshape(y1 - y0, s, x1 - x0, s).mean(axis=(1, 3))
        window = coverage[y0:y1, x0:x1]
        np.maximum(window, local, out=window)

    return (coverage >= threshold).astype(np.uint8)


def load_image(path):
    """Grayscale image as float32 in [0, 1]."""
    with Image.open(path) as img:
        array = np.asarray(img.convert("L"), dtype=np.float32)
    return array / 255.0


@dataclass
class Sample:
    name: str
    image: np.ndarray  # float32 (h, w), [0, 1]
    mask: np.ndarray   # uint8 (h, w), 0 | 1

    @property
    def foreground_fraction(self):
        return float(self.mask.mean())


def load_samples(image_dir, label_dir, supersample=4, verbose=True):
    """Load every labelled image with its rasterised mask."""
    pairs, unlabelled = find_pairs(image_dir, label_dir)
    if not pairs:
        raise FileNotFoundError(
            f"no image/label pairs found between {image_dir} and {label_dir}"
        )
    if unlabelled and verbose:
        names = ", ".join(p.name for p in unlabelled)
        print(f"skipping {len(unlabelled)} unlabelled image(s): {names}")

    samples = []
    for image_path, label_path in pairs:
        image = load_image(image_path)
        circles = load_circles(label_path)
        mask = rasterize_circles(image.shape, circles, supersample=supersample)
        samples.append(Sample(name=image_path.stem, image=image, mask=mask))
        if verbose:
            print(
                f"  {image_path.name}: {image.shape[1]}x{image.shape[0]} px, "
                f"{len(circles)} circles, {mask.mean():.1%} foreground"
            )
    return samples


# ---------------------------------------------------------------------- splits


def split_samples(samples, val_fraction=0.2, test_fraction=0.2, seed=42):
    """Split *by image* (not by patch), so no image appears in two splits."""
    if len(samples) < 3:
        raise ValueError(
            f"need at least 3 labelled images to make train/val/test splits, got {len(samples)}"
        )
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)

    n = len(order)
    n_val = max(1, round(n * val_fraction))
    n_test = max(1, round(n * test_fraction))
    if n_val + n_test >= n:  # always leave at least one training image
        n_val = n_test = 1
    n_train = n - n_val - n_test

    picked = [samples[i] for i in order]
    return {
        "train": picked[:n_train],
        "val": picked[n_train:n_train + n_val],
        "test": picked[n_train + n_val:],
    }


def dataset_statistics(samples):
    """Mean/std intensity and positive-class weight from the training split."""
    pixels = np.concatenate([s.image.ravel() for s in samples])
    masks = np.concatenate([s.mask.ravel() for s in samples])
    positive = float(masks.mean())
    # pos_weight for BCE: ratio of negative to positive pixels
    pos_weight = (1.0 - positive) / positive if positive > 0 else 1.0
    return {
        "mean": float(pixels.mean()),
        "std": float(pixels.std()) or 1.0,
        "foreground_fraction": positive,
        "pos_weight": float(pos_weight),
    }


# -------------------------------------------------------------------- datasets


def pad_to_multiple(tensor, multiple):
    """Pad the last two dims up to a multiple of `multiple` (reflect padding)."""
    h, w = tensor.shape[-2:]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (h, w)
    padded = torch.nn.functional.pad(
        tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect"
    ).squeeze(0)
    return padded, (h, w)


class PatchDataset(Dataset):
    """Random augmented crops from the training images.

    One "epoch" is `samples_per_epoch` random crops rather than a pass over the
    handful of source images, which decouples epoch length from dataset size.
    """

    def __init__(
        self,
        samples,
        patch_size=128,
        samples_per_epoch=200,
        mean=0.0,
        std=1.0,
        augment=True,
        seed=0,
    ):
        self.samples = samples
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.mean = mean
        self.std = std
        self.augment = augment
        self.seed = seed
        self.epoch = 0

        for sample in samples:
            h, w = sample.image.shape
            if h < patch_size or w < patch_size:
                raise ValueError(
                    f"{sample.name} is {w}x{h} px, smaller than patch size {patch_size}"
                )

    def set_epoch(self, epoch):
        """Re-seed so each epoch draws different crops but runs stay reproducible."""
        self.epoch = epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        rng = np.random.default_rng((self.seed, self.epoch, index))
        sample = self.samples[rng.integers(len(self.samples))]
        image, mask = sample.image, sample.mask

        h, w = image.shape
        top = int(rng.integers(h - self.patch_size + 1))
        left = int(rng.integers(w - self.patch_size + 1))
        image = image[top:top + self.patch_size, left:left + self.patch_size]
        mask = mask[top:top + self.patch_size, left:left + self.patch_size]

        if self.augment:
            image, mask = self._augment(image, mask, rng)

        image = (image - self.mean) / self.std
        return (
            torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0),
            torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)).unsqueeze(0),
        )

    def _augment(self, image, mask, rng):
        # Dihedral group: EM sections have no preferred orientation.
        k = int(rng.integers(4))
        if k:
            image, mask = np.rot90(image, k), np.rot90(mask, k)
        if rng.random() < 0.5:
            image, mask = np.fliplr(image), np.fliplr(mask)
        if rng.random() < 0.5:
            image, mask = np.flipud(image), np.flipud(mask)

        # Intensity jitter: section-to-section staining/contrast varies.
        image = image.astype(np.float32)
        if rng.random() < 0.5:
            image = image * rng.uniform(0.85, 1.15) + rng.uniform(-0.1, 0.1)
        if rng.random() < 0.3:
            image = image + rng.normal(0.0, 0.02, size=image.shape).astype(np.float32)
        return np.clip(image, 0.0, 1.0), mask


class FullImageDataset(Dataset):
    """Whole images, padded to a size the network can down/upsample cleanly."""

    def __init__(self, samples, mean=0.0, std=1.0, multiple=8):
        self.samples = samples
        self.mean = mean
        self.std = std
        self.multiple = multiple

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = (sample.image - self.mean) / self.std
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(sample.mask.astype(np.float32)).unsqueeze(0)
        image, _ = pad_to_multiple(image, self.multiple)
        mask, _ = pad_to_multiple(mask, self.multiple)
        return image, mask
