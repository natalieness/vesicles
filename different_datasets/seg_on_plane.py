# hemibrain dataset from neuroglancer

from cloudvolume import CloudVolume
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.draw import polygon as sk_polygon
import matplotlib.patches as mpatches
import trimesh
from skimage import measure, segmentation, feature, morphology
import pandas as pd

em_clahe_path = 'precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg'
seg_path      = 'precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation'
mito_by_neuron_path = 'precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/mito-objects-grouped'
mito_path = 'precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/mito-objects'

#%% get image stack for area of interest
mip    = 0

cv_seg   = CloudVolume(seg_path,            use_https=True, fill_missing=True)
mito_seg = CloudVolume(mito_path, use_https=True, fill_missing=True)
vol      = CloudVolume(em_clahe_path,       use_https=True, mip=mip, progress=True, fill_missing=False)

def raw_bbox_to_index_for_em(raw_bbox, mip=0):
    bbox = np.array(raw_bbox, dtype=float)
    # permutate to (z0, z1, y0, y1, x0, x1) and convert from nm to mip0 voxels (8 nm)
    bbox = bbox[[4, 5, 2, 3, 0, 1]]
    bbox = [b // 2**mip for b in bbox]
    return bbox

def plot_em_2d(img, min_percentile=0.5, max_percentile=99.5, z_frame=0):
    fig, ax = plt.subplots(figsize=(10, 10))
    vmin, vmax = np.percentile(img[...,z_frame], (min_percentile, max_percentile))
    ax.imshow(img[...,z_frame], cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')

xy_dist = 500
z_depth = 10
x0, y0, z0 = 28400, 31600, 20966
bbox = [x0, x0 + xy_dist, y0, y0 + xy_dist, z0, z0 + z_depth]
bboxf = raw_bbox_to_index_for_em(bbox, mip=mip)

raw = vol[bboxf[0]:bboxf[1], bboxf[2]:bboxf[3], bboxf[4]:bboxf[5], 0]
print(f"Downloaded shape: {raw.shape}")
img = np.squeeze(raw).T   # (rows=y, cols=x) for imshow

# plot example frame
plot_em_2d(img, z_frame=0)

#%% subtract mito's 
def vol_to_mito_translation(bbox):
    x_shift = 0
    y_shift = 0
    z_shift = 0
    return [bbox[0] + x_shift, bbox[1] + x_shift, bbox[2] + y_shift, bbox[3] + y_shift, bbox[4] + z_shift, bbox[5] + z_shift]

bboxm = vol_to_mito_translation(bboxf)
mito_mask = mito_seg[bboxm[0]:bboxm[1], bboxm[2]:bboxm[3], bboxm[4]:bboxm[5], 0].squeeze().astype(bool)
print(np.sum(mito_mask))
mito_mask = np.transpose(mito_mask, (2, 1, 0))  # check that we have some mito pixels in this bbox
img_no_mito = np.where(mito_mask, 255, img)
plot_em_2d(img_no_mito, z_frame=1)

# %%

for z in range(10):
    plot_em_2d(img_no_mito, z_frame=z)
# %%

import numpy as np
from scipy import ndimage as ndi

def segmentation_boundary_mask(seg, only_between_nonzero=True, thickness=1):
    """
    seg: 3D neuron segmentation, shape (x, y, z)
    Returns boolean mask where neighboring voxels have different labels.

    only_between_nonzero=True:
        keeps only boundaries between two labeled neurons.
        ignores label-vs-background boundaries.

    thickness:
        dilation radius in voxels after finding boundaries.
    """
    seg = np.asarray(seg)
    boundary = np.zeros(seg.shape, dtype=bool)

    for axis in range(3):
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3

        sl_a[axis] = slice(1, None)
        sl_b[axis] = slice(None, -1)

        a = seg[tuple(sl_a)]
        b = seg[tuple(sl_b)]

        diff = a != b

        if only_between_nonzero:
            diff &= (a != 0) & (b != 0)

        # Mark both sides of the boundary
        boundary[tuple(sl_a)] |= diff
        boundary[tuple(sl_b)] |= diff

    if thickness > 1:
        boundary = ndi.binary_dilation(boundary, iterations=thickness - 1)

    return boundary

def show_boundary_overlay(em_xyz, boundary_xyz, z_frame=0):
    fig, ax = plt.subplots(figsize=(8, 8))

    em2d = em_xyz[:, :, z_frame]
    b2d = boundary_xyz[:, :, z_frame]

    vmin, vmax = np.percentile(em2d, (0.5, 99.5))
    ax.imshow(em2d, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    ax.imshow(
        np.ma.masked_where(~b2d, b2d),
        alpha=0.45,
        interpolation="nearest",
    )

    ax.set_axis_off()
    ax.set_title(f"Boundary overlay, z={z_frame}")
    return fig, ax


# em: shape (x, y, z)
# neuron_seg: shape (x, y, z)

membrane_mask = segmentation_boundary_mask(
    cv_seg[bboxf[0]:bboxf[1], bboxf[2]:bboxf[3], bboxf[4]:bboxf[5], 0],
    only_between_nonzero=True,
    thickness=2,
)
membrane_mask = np.transpose(membrane_mask[...,0], (2, 1, 0))  # check that we have some boundaries in this bbox

em_no_boundaries = img_no_mito.copy()
em_no_boundaries[membrane_mask] = 255

show_boundary_overlay(em_no_boundaries, membrane_mask, z_frame=0)
plot_em_2d(em_no_boundaries, z_frame=0)
# %%


cutout = em_no_boundaries.copy()   # (x, y, z) = (300, 300, 10)

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

THRESHOLD       = 180       # 0–255
THRESHOLD_MODE  = 'below'   # 'below' | 'above'
MIN_VOXELS      = 5        # voxels; remove small noise blobs
USE_WATERSHED   = True      # True = watershed | False = connected-components
WS_MIN_DIST     = 1        # px between watershed seeds
WS_SMOOTH_SIGMA = 1.0       # Gaussian blur on distance map before peak finding
                             #   ↑ higher → softer/fewer splits (try 1–5)
                             #   0 = no smoothing (harshest)
WS_COMPACTNESS  = 0.001     # 0 = classic | higher = more uniform seeds

MAX_VOLUME      = 1000      # voxels; drop objects larger than this  (None = off)
MIN_SOLIDITY    = 0.3     # 0–1; drop objects below this solidity  (None = off)

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
rgba_lut = get_rgba_lut(n_objects)
fig = plot_segmentation(labels_3d, volume, rgba_lut, n_objects, THRESHOLD, THRESHOLD_MODE, method_str, nz, single_z=0)

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

n_objects = int(labels_3d.max())
rgba_lut = get_rgba_lut(n_objects)
fig = plot_segmentation(labels_3d, volume, rgba_lut, n_objects, THRESHOLD, THRESHOLD_MODE, method_str, nz, single_z=0)
# %%

# ── Colour LUT  (label 0 = background → transparent) ─────────────────────────
def get_rgba_lut(n_objects):
    cmap_src  = plt.colormaps['tab20']
    rgba_lut  = np.zeros((n_objects + 1, 4), dtype=float)
    for i in range(1, n_objects + 1):
        rgba_lut[i] = cmap_src((i - 1) % 20)
    return rgba_lut

# ── Plot function: visualize current objects/masks ────────────────────────────
def plot_segmentation(labels_3d, volume, rgba_lut, n_objects, THRESHOLD, THRESHOLD_MODE, 
                      method_str, nz, single_z=None):
    """
    Plot 3D segmentation results with coloured object overlays.
    
    Parameters:
    -----------
    labels_3d : ndarray, shape (nz, ny, nx)
        3D labeled segmentation (label 0 = background)
    volume : ndarray, shape (nz, ny, nx)
        EM volume data
    rgba_lut : ndarray, shape (n_objects+1, 4)
        RGBA color lookup table for each label
    n_objects : int
        Number of segmented objects
    THRESHOLD : int
        Threshold value used for segmentation (0-255)
    THRESHOLD_MODE : str
        'below' or 'above'
    method_str : str
        Description of segmentation method
    nz : int
        Number of z-slices in volume
    single_z : int or None, optional
        If None (default): plot all z-slices
        If int: plot only the specified z-slice (0-indexed)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    vmin_v = float(np.percentile(volume, 0.5))
    vmax_v = float(np.percentile(volume, 99.5))
    
    if single_z is not None:
        # Plot single z-slice
        if not (0 <= single_z < nz):
            print(f"Error: single_z={single_z} out of range [0, {nz-1}]")
            return None
        
        fig, ax = plt.subplots(figsize=(5, 5))
        z = single_z
        
        ax.imshow(volume[z], cmap='gray', vmin=vmin_v, vmax=vmax_v,
                  interpolation='nearest')

        sl = labels_3d[z]
        if sl.max() > 0:
            # Semi-transparent coloured overlay only — no text labels
            overlay         = rgba_lut[sl].copy()        # (ny, nx, 4)
            overlay[..., 3] = np.where(sl > 0, 0.85, 0.0)
            ax.imshow(overlay, interpolation='nearest')

        ax.set_title(f'z = {z}', fontsize=10)
        ax.axis('off')
        
        fig.suptitle(
            f'3D segmentation (z={z})  |  thresh={THRESHOLD} ({THRESHOLD_MODE})  '
            f'|  {method_str}  |  {n_objects} objects',
            fontsize=10,
        )
    else:
        # Plot all z-slices
        ncols     = 2
        nrows     = int(np.ceil(nz / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.6, nrows * 3.6 + 1.0),
                                 squeeze=False)
        axes_flat = axes.flatten()

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
    
    return fig


# ── Call plotting function ────────────────────────────────────────────────────
# Plot all z-slices (default behavior)
fig = plot_segmentation(labels_3d, volume, rgba_lut, n_objects, THRESHOLD, THRESHOLD_MODE, 
                  method_str, nz)


# %%
