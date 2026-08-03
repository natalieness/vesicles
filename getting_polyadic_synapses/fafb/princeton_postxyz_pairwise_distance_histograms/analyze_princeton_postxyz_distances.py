#!/usr/bin/env python3
"""Analyze all unique pairwise post_xyz distances for five high-output FAFB neurons.

The source is the Princeton v783 CSV.GZ table.  The script streams the table
twice (counts, then selected coordinates), never creates an n-by-n distance
matrix, and calculates the condensed all-unique-pairs vector in bounded blocks.
"""

from __future__ import annotations

import argparse
import gc
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT_COL = "pre_root_id_720575940"
COORD_COLS = ["post_x", "post_y", "post_z"]
ROOT_PREFIX = "720575940"
N_SELECT = 5
SELECTION_CAP = 10_000
CHUNK_ROWS = 2_000_000
SHORT_MAX_NM = 500.0
SHORT_BIN_WIDTH_NM = 10.0
BROAD_BIN_WIDTH_NM = 5_000.0
REFERENCE_LINES_NM = [100.0, 150.0, 200.0, 250.0]
LOW_QUANTILES = [0.0001, 0.001, 0.005, 0.01, 0.05]
DISTANCE_BLOCK_ROWS = 768


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_source = here.parent / "fafb_data" / "fafb_v783_princeton_synapse_table.csv.gz"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path, default=here)
    parser.add_argument("--selection-cap", type=int, default=SELECTION_CAP)
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    return parser.parse_args()


def full_root_id(short_id: int) -> str:
    """Reconstruct the full FlyWire ID represented by this prefixed column."""
    return f"{ROOT_PREFIX}{short_id:09d}"


def count_presynaptic_rows(
    source: Path, chunk_rows: int
) -> tuple[Counter, dict[str, int]]:
    counts: Counter = Counter()
    qc = {
        "table_rows": 0,
        "missing_or_nonnumeric_root_rows": 0,
        "nonpositive_root_rows": 0,
    }
    for chunk in pd.read_csv(
        source,
        compression="gzip",
        usecols=[ROOT_COL],
        chunksize=chunk_rows,
    ):
        roots = pd.to_numeric(chunk[ROOT_COL], errors="coerce")
        qc["table_rows"] += len(chunk)
        qc["missing_or_nonnumeric_root_rows"] += int(roots.isna().sum())
        valid_numeric = roots.dropna()
        qc["nonpositive_root_rows"] += int((valid_numeric <= 0).sum())
        valid = valid_numeric[valid_numeric > 0].astype(np.int64)
        values = valid.value_counts(sort=False)
        counts.update({int(k): int(v) for k, v in values.items()})
    return counts, qc


def select_neurons(
    counts: Counter, selection_cap: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[int, int]]:
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    rank_by_id = {root_id: rank for rank, (root_id, _) in enumerate(ranked, start=1)}
    eligible = [item for item in ranked if item[1] <= selection_cap]
    if len(eligible) < N_SELECT:
        raise RuntimeError(
            f"Only {len(eligible)} neurons have counts <= {selection_cap}; "
            f"cannot select {N_SELECT}."
        )
    return eligible[:N_SELECT], ranked[:N_SELECT], rank_by_id


def extract_selected_coordinates(
    source: Path, selected_ids: set[int], chunk_rows: int
) -> tuple[dict[int, np.ndarray], dict[str, int], dict[int, int]]:
    parts: dict[int, list[np.ndarray]] = {root_id: [] for root_id in selected_ids}
    selected_invalid_coords = {root_id: 0 for root_id in selected_ids}
    qc = {
        "rows_scanned_for_coordinates": 0,
        "rows_with_any_missing_or_nonnumeric_post_coordinate": 0,
    }

    for chunk in pd.read_csv(
        source,
        compression="gzip",
        usecols=[ROOT_COL, *COORD_COLS],
        chunksize=chunk_rows,
    ):
        qc["rows_scanned_for_coordinates"] += len(chunk)
        coords_numeric = chunk[COORD_COLS].apply(pd.to_numeric, errors="coerce")
        coord_valid = np.isfinite(coords_numeric.to_numpy(dtype=np.float64)).all(axis=1)
        qc["rows_with_any_missing_or_nonnumeric_post_coordinate"] += int(
            (~coord_valid).sum()
        )

        roots = pd.to_numeric(chunk[ROOT_COL], errors="coerce")
        selected_mask = roots.isin(selected_ids).to_numpy()
        if not selected_mask.any():
            continue

        for root_id in selected_ids:
            root_mask = selected_mask & roots.eq(root_id).to_numpy()
            selected_invalid_coords[root_id] += int((root_mask & ~coord_valid).sum())
            usable = root_mask & coord_valid
            if usable.any():
                parts[root_id].append(
                    coords_numeric.loc[usable, COORD_COLS].to_numpy(
                        dtype=np.float64, copy=True
                    )
                )

    coords_by_id = {
        root_id: (
            np.concatenate(root_parts, axis=0)
            if root_parts
            else np.empty((0, 3), dtype=np.float64)
        )
        for root_id, root_parts in parts.items()
    }
    return coords_by_id, qc, selected_invalid_coords


def duplicate_coordinate_stats(coords: np.ndarray) -> dict[str, int]:
    _, multiplicities = np.unique(coords, axis=0, return_counts=True)
    repeated = multiplicities[multiplicities > 1]
    return {
        "unique_post_xyz_count": int(len(multiplicities)),
        "duplicate_coordinate_locations": int(len(repeated)),
        "duplicate_coordinate_rows_beyond_first": int(np.sum(repeated - 1)),
        "exact_duplicate_coordinate_pairs": int(
            np.sum(repeated * (repeated - 1) // 2)
        ),
    }


def condensed_pairwise_distances(
    coords: np.ndarray, block_rows: int = DISTANCE_BLOCK_ROWS
) -> np.ndarray:
    """Return all i<j Euclidean distances without constructing an n-by-n matrix."""
    n = len(coords)
    out = np.empty(n * (n - 1) // 2, dtype=np.float64)
    cursor = 0
    for i0 in range(0, n, block_rows):
        i1 = min(i0 + block_rows, n)
        a = coords[i0:i1]
        a_sq = np.einsum("ij,ij->i", a, a)

        # Unique pairs inside this diagonal block.
        d2 = a_sq[:, None] + a_sq[None, :] - 2.0 * (a @ a.T)
        np.maximum(d2, 0.0, out=d2)
        tri = np.triu_indices(len(a), k=1)
        within = np.sqrt(d2[tri])
        out[cursor : cursor + within.size] = within
        cursor += within.size
        del d2, within

        # Every pair between this block and each later block.
        for j0 in range(i1, n, block_rows):
            j1 = min(j0 + block_rows, n)
            b = coords[j0:j1]
            b_sq = np.einsum("ij,ij->i", b, b)
            d2 = a_sq[:, None] + b_sq[None, :] - 2.0 * (a @ b.T)
            np.maximum(d2, 0.0, out=d2)
            np.sqrt(d2, out=d2)
            flat = d2.ravel()
            out[cursor : cursor + flat.size] = flat
            cursor += flat.size
            del d2, flat
    if cursor != out.size:
        raise AssertionError(f"Filled {cursor} distances, expected {out.size}")
    return out


def interpret_short_distances(row: dict[str, object]) -> str:
    zero_pairs = int(row["exact_duplicate_coordinate_pairs"])
    near = int(row["pairs_le_250_nm"])
    outer = int(row["pairs_gt_250_le_500_nm"])
    peak_left = float(row["short_hist_peak_bin_left_nm"])
    ratio = near / max(outer, 1)
    duplicate_phrase = (
        f"{zero_pairs:,} exact-zero pairs" if zero_pairs else "no exact-zero pairs"
    )
    if ratio >= 1.5 and peak_left < 150:
        finding = (
            f"Strong short-distance enrichment: the 0-250 nm band has {ratio:.2f}x "
            f"as many pairs as >250-500 nm, and the largest 10-nm bin is "
            f"{peak_left:.0f}-{peak_left + SHORT_BIN_WIDTH_NM:.0f} nm"
        )
    elif ratio >= 0.9 and peak_left < 150:
        finding = (
            f"Modest short-distance peak: the largest 10-nm bin is "
            f"{peak_left:.0f}-{peak_left + SHORT_BIN_WIDTH_NM:.0f} nm and the "
            f"0-250 nm band has {ratio:.2f}x the count in >250-500 nm"
        )
    elif peak_left < 150:
        finding = (
            f"A visible local short-distance peak occurs at "
            f"{peak_left:.0f}-{peak_left + SHORT_BIN_WIDTH_NM:.0f} nm, but the "
            f"aggregate 0-250 nm count is only {ratio:.2f}x the >250-500 nm count, "
            "so broad short-range enrichment is not clear"
        )
    else:
        finding = (
            f"No distinct early short-distance peak is apparent: the largest "
            f"10-nm bin in 0-500 nm is {peak_left:.0f}-"
            f"{peak_left + SHORT_BIN_WIDTH_NM:.0f} nm, and the 0-250 nm band has "
            f"{ratio:.2f}x the >250-500 nm count"
        )
    return (
        f"{finding}; {duplicate_phrase} were retained. "
        "The pattern is descriptive and may be consistent with nearby rows sharing a "
        "predicted presynaptic site, but it is not a formal polyad assignment."
    )


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def _compact_count(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def _draw_hist_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    edges: np.ndarray,
    counts: np.ndarray,
    title_lines: list[str],
    x_label: str,
    y_label: str,
    color: str,
    log_y: bool,
    reference_lines: list[float] | None = None,
    show_reference_labels: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    title_font = _font(23, bold=True)
    axis_font = _font(19)
    tick_font = _font(16)
    small_font = _font(14)
    left, right, top, bottom = 105, 28, 82, 70
    px0, py0, px1, py1 = x0 + left, y0 + top, x1 - right, y1 - bottom
    ink, grid = "#20242B", "#D9DEE5"

    draw.text((x0 + 8, y0 + 2), title_lines[0], font=title_font, fill=ink)
    draw.text((x0 + 8, y0 + 31), title_lines[1], font=axis_font, fill="#4B5563")

    values = np.asarray(counts, dtype=np.float64)
    if log_y:
        transformed = np.log10(values + 1.0)
        ymax = max(1.0, float(transformed.max()) * 1.05)
        tick_raw = np.geomspace(1, max(10.0, float(values.max())), 5)
        tick_positions = np.log10(tick_raw + 1.0)
    else:
        transformed = values
        ymax = max(1.0, float(values.max()) * 1.05)
        tick_raw = np.linspace(0.0, ymax, 5)
        tick_positions = tick_raw

    for raw, pos in zip(tick_raw, tick_positions):
        yy = py1 - int((pos / ymax) * (py1 - py0))
        draw.line((px0, yy, px1, yy), fill=grid, width=1)
        label = _compact_count(raw)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((px0 - 10 - (bbox[2] - bbox[0]), yy - 9), label, font=tick_font, fill=ink)

    xmin, xmax = float(edges[0]), float(edges[-1])
    for left_edge, right_edge, height in zip(edges[:-1], edges[1:], transformed):
        bx0 = px0 + int((left_edge - xmin) / (xmax - xmin) * (px1 - px0))
        bx1 = px0 + int((right_edge - xmin) / (xmax - xmin) * (px1 - px0))
        by = py1 - int((height / ymax) * (py1 - py0))
        if bx1 <= bx0:
            bx1 = bx0 + 1
        draw.rectangle((bx0, by, bx1, py1), fill=color)

    if reference_lines:
        line_colors = ["#6B7280", "#9CA3AF", "#4B5563", "#111827"]
        for ref, line_color in zip(reference_lines, line_colors):
            xx = px0 + int((ref - xmin) / (xmax - xmin) * (px1 - px0))
            draw.line((xx, py0, xx, py1), fill=line_color, width=3)
            if show_reference_labels:
                draw.text((xx + 4, py0 + 4), f"{int(ref)}", font=small_font, fill=line_color)

    draw.line((px0, py0, px0, py1), fill=ink, width=2)
    draw.line((px0, py1, px1, py1), fill=ink, width=2)
    for fraction in np.linspace(0, 1, 6):
        xx = px0 + int(fraction * (px1 - px0))
        val = xmin + fraction * (xmax - xmin)
        label = f"{val:.0f}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.line((xx, py1, xx, py1 + 6), fill=ink, width=2)
        draw.text((xx - (bbox[2] - bbox[0]) / 2, py1 + 9), label, font=tick_font, fill=ink)

    bbox = draw.textbbox((0, 0), x_label, font=axis_font)
    draw.text(
        ((px0 + px1 - (bbox[2] - bbox[0])) / 2, y1 - 29),
        x_label,
        font=axis_font,
        fill=ink,
    )
    # Pillow supports anchored rotated text most reliably via a transparent layer.
    label_layer = Image.new("RGBA", (400, 50), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_layer)
    label_draw.text((0, 0), y_label, font=axis_font, fill=ink)
    rotated = label_layer.crop(label_layer.getbbox()).rotate(90, expand=True)
    panel_image = draw._image
    panel_image.alpha_composite(
        rotated,
        (x0 + 12, int((py0 + py1 - rotated.height) / 2)),
    )


def render_histogram_figure(
    summary_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    selected_order: list[int],
    broad_edges_nm: np.ndarray,
    short_edges_nm: np.ndarray,
    output_png: Path,
    output_pdf: Path,
) -> None:
    width, height = 2600, 3650
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(34, bold=True)
    subtitle_font = _font(23)
    draw.text(
        (60, 30),
        "FAFB v783 Princeton table: all unique pairwise distances between outgoing post_xyz rows",
        font=title_font,
        fill="#18212C",
    )
    draw.text(
        (60, 76),
        "Coordinates are native FlyWire-space nanometres; no unit conversion applied",
        font=subtitle_font,
        fill="#4B5563",
    )

    panel_top, panel_height = 135, 690
    panel_width, gutter = 1240, 35
    colors = ["#2864A5", "#0E8A6A", "#A95C15", "#7B4AB5", "#B53A54"]
    for row_index, root_id in enumerate(selected_order):
        row = summary_df.loc[summary_df["neuron_id_as_stored"].eq(root_id)].iloc[0]
        broad = hist_df[
            hist_df["neuron_id_as_stored"].eq(root_id)
            & hist_df["view"].eq("broad_full_range_5um_bins")
        ]
        short = hist_df[
            hist_df["neuron_id_as_stored"].eq(root_id)
            & hist_df["view"].eq("short_0_500nm_10nm_bins")
        ]
        y0 = panel_top + row_index * panel_height
        max_um = math.ceil(float(row["maximum_distance_nm"]) / 5_000) * 5
        broad_mask = broad_edges_nm[:-1] / 1000.0 < max_um
        broad_edges_plot = np.append(
            broad_edges_nm[:-1][broad_mask] / 1000.0, max_um
        )
        broad_counts_plot = broad["pair_count"].to_numpy()[broad_mask]
        _draw_hist_panel(
            draw,
            (45, y0, 45 + panel_width, y0 + panel_height - 25),
            broad_edges_plot,
            broad_counts_plot,
            [
                f"Neuron {root_id} — full range",
                f"n={int(row['coordinate_rows_used']):,}; pairs={int(row['number_of_pairwise_distances']):,}",
            ],
            "Pairwise distance (µm)",
            "Unique pair count",
            colors[row_index],
            log_y=False,
        )
        _draw_hist_panel(
            draw,
            (
                45 + panel_width + gutter,
                y0,
                45 + 2 * panel_width + gutter,
                y0 + panel_height - 25,
            ),
            short_edges_nm,
            short["pair_count"].to_numpy(),
            [
                f"Neuron {root_id} — short-distance zoom",
                f"≤250 nm: {int(row['pairs_le_250_nm']):,}; exact-zero pairs: {int(row['exact_duplicate_coordinate_pairs']):,}",
            ],
            "Pairwise distance (nm)",
            "Unique pair count (log scale)",
            colors[row_index],
            log_y=True,
            reference_lines=REFERENCE_LINES_NM,
            show_reference_labels=row_index == 0,
        )

    rgb = image.convert("RGB")
    rgb.save(output_png, format="PNG", dpi=(240, 240), optimize=True)
    rgb.save(output_pdf, format="PDF", resolution=180.0)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(source)

    counts, count_qc = count_presynaptic_rows(source, args.chunk_rows)
    selected, unrestricted_top_five, rank_by_id = select_neurons(
        counts, args.selection_cap
    )
    selected_ids = {root_id for root_id, _ in selected}

    coords_by_id, coord_qc, selected_invalid_coords = extract_selected_coordinates(
        source, selected_ids, args.chunk_rows
    )

    count_values = np.fromiter(counts.values(), dtype=np.int64)
    n_valid_neurons = len(count_values)
    selected_order = [root_id for root_id, _ in selected]

    # All values in the source are already physical FlyWire-space nanometres.
    # No scaling is applied.  The upper bound gives common broad-view bin edges
    # without a preliminary all-pairs pass.
    bound_by_id = {
        root_id: float(
            np.linalg.norm(
                coords_by_id[root_id].max(axis=0) - coords_by_id[root_id].min(axis=0)
            )
        )
        for root_id in selected_order
    }
    broad_upper_nm = max(BROAD_BIN_WIDTH_NM, max(bound_by_id.values()))
    broad_upper_nm = math.ceil(broad_upper_nm / BROAD_BIN_WIDTH_NM) * BROAD_BIN_WIDTH_NM
    broad_edges_nm = np.arange(
        0.0, broad_upper_nm + BROAD_BIN_WIDTH_NM, BROAD_BIN_WIDTH_NM
    )
    short_edges_nm = np.arange(
        0.0, SHORT_MAX_NM + SHORT_BIN_WIDTH_NM, SHORT_BIN_WIDTH_NM
    )

    hist_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for root_id in selected_order:
        table_count = int(counts[root_id])
        coords = coords_by_id[root_id]
        n = len(coords)
        if n < 2:
            raise RuntimeError(f"Neuron {root_id} has fewer than two usable coordinates.")
        expected_pairs = n * (n - 1) // 2

        duplicates = duplicate_coordinate_stats(coords)
        distances_nm = condensed_pairwise_distances(coords)
        if distances_nm.size != expected_pairs:
            raise AssertionError("pdist length does not equal n choose 2")

        broad_counts, _ = np.histogram(distances_nm, bins=broad_edges_nm)
        short_counts, _ = np.histogram(distances_nm, bins=short_edges_nm)

        min_nm = float(distances_nm.min())
        max_nm = float(distances_nm.max())
        mean_nm = float(distances_nm.mean(dtype=np.float64))
        threshold_counts = {
            threshold: int(np.count_nonzero(distances_nm <= threshold))
            for threshold in REFERENCE_LINES_NM + [SHORT_MAX_NM]
        }
        count_gt_250_le_500 = int(
            np.count_nonzero(
                (distances_nm > 250.0) & (distances_nm <= SHORT_MAX_NM)
            )
        )
        quantile_values = np.quantile(
            distances_nm,
            LOW_QUANTILES + [0.5],
            method="linear",
            overwrite_input=True,
        )
        del distances_nm
        gc.collect()

        peak_index = int(np.argmax(short_counts))
        summary = {
            "neuron_id_as_stored": root_id,
            "full_flywire_root_id": full_root_id(root_id),
            "outgoing_synapse_count_table": table_count,
            "coordinate_rows_used": n,
            "selected_count_rank": rank_by_id[root_id],
            "percentile_of_synapse_count": float(
                100.0 * np.count_nonzero(count_values <= table_count) / n_valid_neurons
            ),
            "number_of_pairwise_distances": expected_pairs,
            "minimum_distance_nm": min_nm,
            "q0_01_percent_distance_nm": float(quantile_values[0]),
            "q0_1_percent_distance_nm": float(quantile_values[1]),
            "q0_5_percent_distance_nm": float(quantile_values[2]),
            "q1_percent_distance_nm": float(quantile_values[3]),
            "q5_percent_distance_nm": float(quantile_values[4]),
            "median_distance_nm": float(quantile_values[5]),
            "mean_distance_nm": mean_nm,
            "maximum_distance_nm": max_nm,
            "pairs_le_100_nm": threshold_counts[100.0],
            "pairs_le_150_nm": threshold_counts[150.0],
            "pairs_le_200_nm": threshold_counts[200.0],
            "pairs_le_250_nm": threshold_counts[250.0],
            "pairs_le_500_nm": threshold_counts[500.0],
            "pairs_gt_250_le_500_nm": count_gt_250_le_500,
            "fraction_pairs_le_100_nm": threshold_counts[100.0] / expected_pairs,
            "fraction_pairs_le_150_nm": threshold_counts[150.0] / expected_pairs,
            "fraction_pairs_le_200_nm": threshold_counts[200.0] / expected_pairs,
            "fraction_pairs_le_250_nm": threshold_counts[250.0] / expected_pairs,
            "fraction_pairs_le_500_nm": threshold_counts[500.0] / expected_pairs,
            **duplicates,
            "invalid_coordinate_rows_dropped": selected_invalid_coords[root_id],
            "short_hist_peak_bin_left_nm": float(short_edges_nm[peak_index]),
            "short_hist_peak_bin_right_nm": float(short_edges_nm[peak_index + 1]),
            "coordinate_unit": "nm",
            "coordinate_conversion_applied": "none; source coordinates already nm",
        }
        summary["short_distance_interpretation"] = interpret_short_distances(summary)
        summary_rows.append(summary)

        for view, edges, bin_counts in [
            ("broad_full_range_5um_bins", broad_edges_nm, broad_counts),
            ("short_0_500nm_10nm_bins", short_edges_nm, short_counts),
        ]:
            for left, right, bin_count in zip(edges[:-1], edges[1:], bin_counts):
                hist_rows.append(
                    {
                        "neuron_id_as_stored": root_id,
                        "full_flywire_root_id": full_root_id(root_id),
                        "view": view,
                        "bin_left_nm": float(left),
                        "bin_right_nm": float(right),
                        "bin_width_nm": float(right - left),
                        "pair_count": int(bin_count),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    hist_df = pd.DataFrame(hist_rows)
    summary_path = output_dir / "selected_neuron_pairwise_distance_summary.csv"
    hist_path = output_dir / "pairwise_distance_histogram_bins.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.9g")
    hist_df.to_csv(hist_path, index=False, float_format="%.9g")

    figure_png = output_dir / "postxyz_pairwise_distance_histograms.png"
    figure_pdf = output_dir / "postxyz_pairwise_distance_histograms.pdf"
    render_histogram_figure(
        summary_df,
        hist_df,
        selected_order,
        broad_edges_nm,
        short_edges_nm,
        figure_png,
        figure_pdf,
    )

    notes_path = output_dir / "analysis_notes.md"
    top_lines = "\n".join(
        f"- `{root_id}`: {count:,} rows, {count * (count - 1) // 2:,} unique pairs"
        for root_id, count in unrestricted_top_five
    )
    selected_lines = "\n".join(
        f"- `{int(row.neuron_id_as_stored)}` "
        f"(full `{row.full_flywire_root_id}`): "
        f"{int(row.outgoing_synapse_count_table):,} outgoing rows; "
        f"{int(row.number_of_pairwise_distances):,} pairs; "
        f"{int(row.duplicate_coordinate_rows_beyond_first):,} duplicate rows beyond first. "
        f"{row.short_distance_interpretation}"
        for row in summary_df.itertuples(index=False)
    )
    notes_path.write_text(
        f"""# Princeton post_xyz pairwise-distance analysis

## Source and schema

- Source: `{source}`
- Table rows: {count_qc['table_rows']:,}
- Distinct valid presynaptic IDs: {len(counts):,}
- Presynaptic ID column: `{ROOT_COL}`
- Postsynaptic coordinate columns: `{', '.join(COORD_COLS)}`
- The ID values in `{ROOT_COL}` are the suffix stored after the common
  `{ROOT_PREFIX}` prefix. Both forms are included in the CSV outputs.

## Coordinate units

The source coordinates were treated as **nanometres (nm)** and no conversion was
applied. Local repository evidence is the analysis file
`getting_polyadic_synapses/fafb_poly.py`, whose comments at the coordinate
extraction identify the table's native coordinate space as "FlyWire nm" and pass
the values to `xform.flywire_to_fafb14(..., coordinates='nm')`. The coordinate
values are already physical magnitudes on the expected anisotropic FlyWire grid;
they are not unscaled voxel indices.

## Selection and computational method

The table's absolute top five IDs were:

{top_lines}

Those neurons would each require approximately 1.7-7.3 billion pairwise
distances. To perform the requested **complete** all-unique-pairs calculation
without an excessive dense matrix, the reproducible selection rule was: rank all
valid positive IDs by outgoing row count and take the five highest counts at or
below {args.selection_cap:,}. These neurons rank {min(rank_by_id[x] for x in selected_order)}
to {max(rank_by_id[x] for x in selected_order)} of {len(counts):,} by count and
are at approximately the {summary_df['percentile_of_synapse_count'].min():.3f}th
to {summary_df['percentile_of_synapse_count'].max():.3f}th percentiles.
This is the same presynaptic-neuron selection rule as the pre_xyz analysis; only
the coordinates used in the distance calculation changed to `post_x`,
`post_y`, and `post_z`.

Distances were computed in bounded NumPy blocks into only the condensed
`n(n-1)/2` unique-distance vector (no self-distances, reverse-order duplicates,
or dense n-by-n matrix). Broad histograms use 5 µm bins and the short-distance
panels use 10 nm bins from 0-500 nm. Summary quantiles are exact linear empirical
quantiles of the condensed distance vector.

## Preprocessing and quality control

- Missing/nonnumeric presynaptic ID rows: {count_qc['missing_or_nonnumeric_root_rows']:,}
- Nonpositive/placeholder presynaptic ID rows: {count_qc['nonpositive_root_rows']:,}
- Rows with any missing/nonnumeric/nonfinite `post_xyz` coordinate: {coord_qc['rows_with_any_missing_or_nonnumeric_post_coordinate']:,}
- Selected rows dropped for invalid coordinates: {sum(selected_invalid_coords.values()):,}
- Exact duplicate coordinates were retained, because repeated `post_xyz` rows can
  be the signal of interest. They produce zero-distance pairs and are reported
  per neuron in the summary CSV.

## Selected neurons and descriptive interpretation

{selected_lines}

The short-range comparisons are descriptive. Without a spatial null model and
independent connector labels, they should not be read as formal evidence that
any specific rows belong to the same biological polyad.
""",
        encoding="utf-8",
    )

    required_outputs = [
        summary_path,
        hist_path,
        figure_png,
        figure_pdf,
        notes_path,
        Path(__file__).resolve(),
    ]
    missing = [str(path) for path in required_outputs if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty outputs: {missing}")

    print(summary_df.to_string(index=False))
    print("\nCreated:")
    for path in required_outputs:
        print(f"{path}\t{path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
