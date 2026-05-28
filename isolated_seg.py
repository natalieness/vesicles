# hemibrain dataset from neuroglancer

from cloudvolume import CloudVolume
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.draw import polygon as sk_polygon
import matplotlib.patches as mpatches
import trimesh

em_clahe_path = 'precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg'
seg_path      = 'precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation'
mito_by_neuron_path = 'precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/mito-objects-grouped'

#%% ── parameters ──────────────────────────────────────────────────────────────
seg_id = 487834179   # KCa'b'-ap1
mip    = 1

# ── load volumes & meshes ────────────────────────────────────────────────────
cv_seg   = CloudVolume(seg_path,            use_https=True, fill_missing=True)
mito_seg = CloudVolume(mito_by_neuron_path, use_https=True, fill_missing=True)
vol      = CloudVolume(em_clahe_path,       use_https=True, mip=mip,
                       progress=True, fill_missing=False)

ex      = cv_seg.mesh.get(seg_id)[seg_id]
ex_mito = mito_seg.mesh.get(seg_id)[seg_id]

# ── build trimesh objects from the 3-D meshes ────────────────────────────────
# Build once, query many times.  process=False preserves the original geometry.
print("Building trimesh objects …")
tm_neuron = trimesh.Trimesh(
    vertices=np.asarray(ex.vertices,      dtype=float),
    faces=np.asarray(ex.faces),
    process=False,
)
tm_mito = trimesh.Trimesh(
    vertices=np.asarray(ex_mito.vertices, dtype=float),
    faces=np.asarray(ex_mito.faces),
    process=False,
)
print(f"  neuron : {len(tm_neuron.vertices):,} verts  "
      f"{len(tm_neuron.faces):,} faces  watertight={tm_neuron.is_watertight}")
print(f"  mito   : {len(tm_mito.vertices):,} verts  "
      f"{len(tm_mito.faces):,} faces  watertight={tm_mito.is_watertight}")

# nm vertex coordinates (for z-slice selection and bounding box)
ex_points      = np.asarray(ex.vertices)
ex_mito_points = np.asarray(ex_mito.vertices)

# ── nm → mip0 voxel coords (8 nm isotropic at mip0) ────────────────────────
ex_points_px      = np.floor(ex_points      / 8.0).astype(int)
ex_mito_points_px = np.floor(ex_mito_points / 8.0).astype(int)

# ── mip0 → current-mip image coords (per-axis scale) ────────────────────────
base_res       = np.array(vol.scales[0]['resolution'][:3], dtype=float)
curr_res       = np.array(vol.scale['resolution'][:3],    dtype=float)
mip_factor_xyz = curr_res / base_res
print(f"\nmip factors xyz @ mip{mip}: {mip_factor_xyz.tolist()}")

ex_points_img      = np.floor(ex_points_px      / mip_factor_xyz).astype(int)
ex_mito_points_img = np.floor(ex_mito_points_px / mip_factor_xyz).astype(int)

# ── choose z-slice: the one with the most neuron vertices ───────────────────
z_vals, z_counts = np.unique(ex_points_img[:, 2], return_counts=True)
best_z = 9880 #int(z_vals[np.argmax(z_counts)])  # trying next one
# Physical z at the voxel centre in nm — used for the trimesh plane cut
z_nm_center = (best_z + 0.5) * curr_res[2]
print(f"\nChosen z = {best_z}  (z_nm = {z_nm_center:.1f} nm)  "
      f"({z_counts.max():,} neuron verts, "
      f"{(ex_mito_points_img[:, 2] == best_z).sum():,} mito verts)")

# ── bounding box of neuron at this z (defines the EM download region) ───────
neu_in_z = ex_points_img[ex_points_img[:, 2] == best_z]
x0 = int(neu_in_z[:, 0].min())
x1 = int(neu_in_z[:, 0].max()) + 1
y0 = int(neu_in_z[:, 1].min())
y1 = int(neu_in_z[:, 1].max()) + 1
print(f"EM patch  x=[{x0},{x1}]  y=[{y0},{y1}]  z={best_z}")

# ── download EM data ─────────────────────────────────────────────────────────
raw = vol[x0:x1, y0:y1, best_z, 0]
print(f"Downloaded shape: {raw.shape}")
img = np.squeeze(raw).T   # (rows=y, cols=x) for imshow

# ── 3-D mesh → 2-D mask ──────────────────────────────────────────────────────
def mesh_cross_section_mask(tm, z_nm, resolution_nm, x0_img, y0_img, shape):
    """
    Cut the 3-D mesh with a horizontal plane at z_nm, rasterise every boundary
    contour onto a 2-D grid, then fill the enclosed regions.

    Why section() + fill rather than tm.contains():
      tm.contains() fires a ray cast per query point — on a patch of thousands of
      voxels against a mesh with hundreds of thousands of faces this takes hours.
      tm.section() intersects all mesh triangles with the plane in O(faces) time,
      returning the exact boundary contours of the solid at that z-level.
      Rasterising those contours and calling binary_fill_holes is equivalent to
      a per-voxel contains() test but completes in seconds.

    Parameters
    ----------
    tm           : trimesh.Trimesh  – full 3-D mesh built before this call
    z_nm         : float  – physical z of the probe plane (nm); use voxel centre
    resolution_nm: (3,)   – [rx, ry, rz] voxel size at current mip (nm)
    x0_img, y0_img: int   – top-left of patch in image-voxel coords
    shape        : (rows, cols) – mask shape  [rows=y, cols=x]

    Returns
    -------
    mask : (rows, cols) bool array  — True inside the mesh at this z
    """
    resolution_nm = np.asarray(resolution_nm, dtype=float)

    # Intersect the 3-D mesh with the z-plane — O(faces), not O(voxels)
    section = tm.section(
        plane_origin=[0.0, 0.0, z_nm],
        plane_normal=[0.0, 0.0, 1.0],
    )

    if section is None:
        print(f"  No intersection found at z_nm={z_nm:.1f}")
        return np.zeros(shape, dtype=bool)

    print(f"  {len(section.entities)} boundary contour(s) at z_nm={z_nm:.1f}")

    # Directly fill each contour polygon.  sk_polygon clips out-of-bounds
    # coords gracefully (unlike polygon_perimeter which crashes on empty clips).
    # Multiple disconnected bodies each become their own entity and are filled
    # independently.
    mask = np.zeros(shape, dtype=bool)

    for entity in section.entities:
        pts = section.vertices[entity.points]          # (N, 3) nm; z ≈ z_nm
        if len(pts) < 3:
            continue
        row = pts[:, 1] / resolution_nm[1] - y0_img
        col = pts[:, 0] / resolution_nm[0] - x0_img
        try:
            rr, cc = sk_polygon(row, col, shape=shape)
            mask[rr, cc] = True
        except Exception as e:
            print(f"  Warning: polygon fill failed ({len(pts)} pts): {e}")

    return mask


# ── build masks ───────────────────────────────────────────────────────────────
print("\nNeuron mask:")
neuron_mask = mesh_cross_section_mask(tm_neuron, z_nm_center, curr_res, x0, y0, img.shape)

print("\nMito mask:")
mito_mask   = mesh_cross_section_mask(tm_mito,   z_nm_center, curr_res, x0, y0, img.shape)

print(f"\nNeuron filled area : {neuron_mask.sum():,} px  "
      f"({neuron_mask.mean()*100:.2f}% of patch)")
print(f"Mito   filled area : {mito_mask.sum():,} px  "
      f"({mito_mask.sum()/max(neuron_mask.sum(),1)*100:.2f}% of neuron area)")

_, n_mito = ndi.label(mito_mask)
print(f"Distinct mito regions in this slice: {n_mito}")

#%% ── plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 12))
fr = [0, -1, 0, -1]   # [x0, x1, y0, y1] crop within patch for display

vmin, vmax = np.percentile(img, (0.5, 99.5))
ax.imshow(img[fr[2]:fr[3], fr[0]:fr[1]],
          cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')

neu_rgba = np.zeros((*img.shape, 4), dtype=float)
neu_rgba[neuron_mask] = [1.0, 0.2, 0.2, 0.20]
ax.imshow(neu_rgba[fr[2]:fr[3], fr[0]:fr[1]], interpolation='nearest')

mito_rgba = np.zeros((*img.shape, 4), dtype=float)
mito_rgba[mito_mask] = [0.15, 0.45, 1.0, 0.55]
ax.imshow(mito_rgba[fr[2]:fr[3], fr[0]:fr[1]], interpolation='nearest')

legend_patches = [
    mpatches.Patch(facecolor=(1.0, 0.2, 0.2, 0.5), label=f"Neuron {seg_id}"),
    mpatches.Patch(facecolor=(0.15, 0.45, 1.0, 0.8),
                   label=f"Mitochondria ({n_mito} bodies)"),
]
ax.legend(handles=legend_patches, loc='upper right', fontsize=12)
ax.set_title(
    f"EM (clahe_yz, mip{mip})  ·  z = {best_z}  ·  seg {seg_id}\n"
    f"z_nm = {z_nm_center:.1f} nm  |  patch x=[{x0},{x1}]  y=[{y0},{y1}]",
    fontsize=11,
)
ax.axis('off')
plt.tight_layout()
plt.show()

# %%
