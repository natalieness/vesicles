# hemibrain dataset from neuroglancer 

from cloudvolume import CloudVolume
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, segmentation, feature, morphology
import matplotlib.patches as mpatches

em_clahe_path = 'precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg'
seg_path = 'precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation'
#%%
# Initialize a CloudVolume for the EM image data
vol = CloudVolume(
    em_clahe_path,
    use_https=True,
    mip=0,
    progress=True,
    fill_missing=False,
)
xy_space = 500
z_space = 10
x0= 19500
y0 = 21610
z0 = 18140
x1, y1, z1 = x0 + xy_space, y0 + xy_space, z0 + z_space

cutout = vol[x0:x1, y0:y1, z0:z1, 0]
# example frame
for zplot in range(cutout.shape[2]):
    fig, ax = plt.subplots(figsize=(10, 10))
    vmin, vmax = np.percentile(cutout[..., zplot, 0], (0.5, 99.5))
    ax.imshow(cutout[..., zplot, 0], cmap='gray', vmin=vmin, vmax=vmax)

# %%
cutout = cutout.squeeze()   # → (x, y, z) = (300, 300, 10)

# %%
# =============================================================================
# 3D SEGMENTATION  ←  edit the parameters block below
# =============================================================================
#
#  THRESHOLD      : pixel intensity (0–255) that defines foreground
#  THRESHOLD_MODE : 'below' = dark objects (e.g. vesicle membranes in raw EM)
#                   'above' = bright objects
#  MIN_VOXELS     : discard any labelled region smaller than this many voxels
#
#  USE_WATERSHED  : True  → watershed split (good for touching/overlapping blobs)
#                   False → plain connected-components (faster, fewer params)
#  WS_MIN_DIST    : minimum voxel distance between watershed seed peaks
#                   ↑ larger → fewer, bigger objects
#                   ↓ smaller → more objects, may over-split
#  WS_COMPACTNESS : 0 = classic geodesic watershed
#                   >0 (e.g. 0.01–1) → seeds grow more uniformly / sphere-like
#                   very high (e.g. 5–20) → nearly Voronoi, ignores intensity
#
#  ── Filtering (applied after segmentation, before plotting) ─────────────────
#  MAX_VOLUME     : drop objects with MORE than this many voxels  (None = off)
#  MIN_SOLIDITY   : (optional) drop objects below this value (0–1)
#                   solidity = voxel_volume / convex_hull_volume
#                   1.0 = perfectly convex (sphere-like)
#                   ~0.8–0.9 keeps compact blobs, rejects irregular/elongated ones
#                   None = no circularity filter
# =============================================================================

THRESHOLD       = 80       # 0–255
THRESHOLD_MODE  = 'below'   # 'below' | 'above'
MIN_VOXELS      = 10        # voxels; remove small noise blobs
USE_WATERSHED   = True      # True = watershed | False = connected-components
WS_MIN_DIST     = 1        # px between watershed seeds
WS_SMOOTH_SIGMA = 10.0       # Gaussian blur on distance map before peak finding
                             #   ↑ higher → softer/fewer splits (try 1–5)
                             #   0 = no smoothing (harshest)
WS_COMPACTNESS  = 0.001     # 0 = classic | higher = more uniform seeds

MAX_VOLUME      = 1000      # voxels; drop objects larger than this  (None = off)
MIN_SOLIDITY    = None      # 0–1; drop objects below this solidity  (None = off)

# =============================================================================



# ── Rearrange to (z, y, x) for natural slice indexing ────────────────────────
volume = np.moveaxis(cutout, -1, 0).astype(np.float32)  # (10, 300, 300)
nz, ny, nx = volume.shape

# ── Binary mask ───────────────────────────────────────────────────────────────
binary = (volume < THRESHOLD) if THRESHOLD_MODE == 'below' else (volume > THRESHOLD)
binary = morphology.remove_small_objects(binary.copy(), min_size=MIN_VOXELS)
binary = morphology.remove_small_holes(binary, area_threshold=MIN_VOXELS)

# ── Label 3D objects ──────────────────────────────────────────────────────────
if USE_WATERSHED:
    dist        = ndi.distance_transform_edt(binary)
    # Smooth the distance map — merges nearby peaks → fewer, softer splits
    dist_smooth = ndi.gaussian_filter(dist, sigma=WS_SMOOTH_SIGMA) if WS_SMOOTH_SIGMA > 0 else dist
    coords      = feature.peak_local_max(dist_smooth, min_distance=WS_MIN_DIST, labels=binary)
    seed_mask   = np.zeros_like(dist, dtype=bool)
    if coords.size:
        seed_mask[tuple(coords.T)] = True
    markers, _  = ndi.label(seed_mask)
    labels_3d   = segmentation.watershed(-dist_smooth, markers, mask=binary,
                                         compactness=WS_COMPACTNESS)
    method_str  = (f'watershed  min_dist={WS_MIN_DIST}  '
                   f'smooth={WS_SMOOTH_SIGMA}  compact={WS_COMPACTNESS}')
else:
    struct     = np.ones((3, 3, 3))          # 26-connectivity
    labels_3d, _ = ndi.label(binary, structure=struct)
    method_str = 'connected-components (26-conn)'

n_objects = int(labels_3d.max())
print(f"\n── Segmentation ──────────────────────────────────────────")
print(f"  method      : {method_str}")
print(f"  threshold   : {THRESHOLD} ({THRESHOLD_MODE})")
print(f"  min voxels  : {MIN_VOXELS}")
print(f"  objects found: {n_objects}\n")
#%%
# ── Per-object dimensions & statistics ───────────────────────────────────────
# (uses regionprops directly to avoid a regionprops_table bug in some skimage
#  versions where 0-d scalar properties cause an IndexError on 3D labels)
if n_objects > 0:
    rows = []
    for r in measure.regionprops(labels_3d, intensity_image=volume):
        z0b, y0b, x0b, z1b, y1b, x1b = r.bbox
        # solidity uses qhull; flat objects (span_z==1) can cause it to fail
        try:
            sol = r.solidity
        except Exception:
            sol = float('nan')
        rows.append({
            'label'          : r.label,
            'area'           : r.area,
            'cen_z'          : r.centroid[0],
            'cen_y'          : r.centroid[1],
            'cen_x'          : r.centroid[2],
            'span_z'         : z1b - z0b,
            'span_y'         : y1b - y0b,
            'span_x'         : x1b - x0b,
            'solidity'       : sol,
            'mean_intensity' : r.mean_intensity,
            'max_intensity'  : r.max_intensity,
        })
    df = pd.DataFrame(rows)
    # Equivalent sphere diameter from voxel volume  Ø = (6V/π)^(1/3)
    df['eq_diam'] = ((6 * df['area'] / np.pi) ** (1 / 3)).round(1)

    show_cols = ['label', 'area', 'span_z', 'span_y', 'span_x',
                 'eq_diam', 'solidity', 'mean_intensity', 'cen_z', 'cen_y', 'cen_x']
    print(df[show_cols].to_string(index=False, float_format='%.2f'))
    print()

#%%
# ── Filter objects ────────────────────────────────────────────────────────────
if n_objects > 0:
    keep = pd.Series(True, index=df.index)
    if MAX_VOLUME is not None:
        keep &= df['area'] <= MAX_VOLUME
    if MIN_SOLIDITY is not None:
        keep &= df['solidity'] >= MIN_SOLIDITY

    n_removed = int((~keep).sum())
    if n_removed > 0:
        orig_max    = int(labels_3d.max())
        drop_labels = df.loc[~keep, 'label'].tolist()
        labels_3d[np.isin(labels_3d, drop_labels)] = 0   # zero out rejected labels

        df = df[keep].reset_index(drop=True)
        n_objects = len(df)

        # Remap remaining label IDs to compact 1..N so colour LUT stays tight
        remap = np.zeros(orig_max + 1, dtype=labels_3d.dtype)
        for new_id, old_id in enumerate(df['label'].tolist(), start=1):
            remap[old_id] = new_id
        labels_3d = remap[labels_3d]
        df['label'] = np.arange(1, n_objects + 1)

        print(f"  removed {n_removed} object(s)  →  {n_objects} remain\n")
        if n_objects > 0:
            print(df[show_cols].to_string(index=False, float_format='%.2f'))
            print()
    else:
        print(f"  no objects removed by filter  ({n_objects} remain)\n")

# ── Colour LUT  (label 0 = background → transparent) ─────────────────────────
cmap_src  = plt.colormaps['tab20']
rgba_lut  = np.zeros((n_objects + 1, 4), dtype=float)
for i in range(1, n_objects + 1):
    rgba_lut[i] = cmap_src((i - 1) % 20)

# ── Plot: 10 z-slices, coloured overlays ─────────────────────────────────────
ncols     = 2
nrows     = int(np.ceil(nz / ncols))
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(ncols * 3.6, nrows * 3.6 + 1.0),
                         squeeze=False)
axes_flat = axes.flatten()

vmin_v = float(np.percentile(volume, 0.5))
vmax_v = float(np.percentile(volume, 99.5))

for z in range(nz):
    ax = axes_flat[z]
    ax.imshow(volume[z], cmap='gray', vmin=vmin_v, vmax=vmax_v,
              interpolation='nearest')

    sl = labels_3d[z]
    if sl.max() > 0:
        # Semi-transparent coloured overlay only — no text labels
        overlay         = rgba_lut[sl].copy()        # (ny, nx, 4)
        overlay[..., 3] = np.where(sl > 0, 0.85, 0.0)
        ax.imshow(overlay, interpolation='nearest')

    ax.set_title(f'z = {z}', fontsize=9)
    ax.axis('off')

# Hide unused subplot panels
for z in range(nz, len(axes_flat)):
    axes_flat[z].set_visible(False)

# Legend
# if n_objects > 0:
#     patches = [mpatches.Patch(facecolor=rgba_lut[i, :3], label=f'obj {i}')
#                for i in range(1, n_objects + 1)]
#     fig.legend(handles=patches, loc='lower center',
#                ncol=min(n_objects, 12), fontsize=8,
#                title='3D objects', framealpha=0.9,
#                bbox_to_anchor=(0.5, -0.01))

fig.suptitle(
    f'3D segmentation  |  thresh={THRESHOLD} ({THRESHOLD_MODE})  '
    f'|  {method_str}  |  {n_objects} objects',
    fontsize=10,
)
plt.tight_layout()
# plt.savefig('segmentation_zslices.png', dpi=150, bbox_inches='tight')
plt.show()
# print("Saved → segmentation_zslices.png")


# %%
