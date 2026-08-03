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
mito_path = 'precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/mito-objects'
mito_by_neuron_path = 'precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/mito-objects-grouped'


#%% get o area of interest 

# first get segmentation of example neuron 
seg_id = 488179878 #KCa'b'-ap1
mip = 1
VIS_DILATION_RADIUS = 2  # set to 0 to keep only exact mesh-vertex pixels
cv_seg = CloudVolume(seg_path, use_https=True, fill_missing=True)
mito_seg = CloudVolume(mito_by_neuron_path, use_https=True, fill_missing=True)
vol = CloudVolume(
    em_clahe_path,
    use_https=True,
    mip=mip,
    progress=True,
    fill_missing=False,
)

ex = cv_seg.mesh.get(seg_id)[seg_id]
ex_mito = mito_seg.mesh.get(seg_id)[seg_id]

# check overlap of mito and neuron
ex_points = ex.vertices
ex_mito_points = ex_mito.vertices

# get bounding box of mito and

ex_mito_mask = np.isin(ex_mito_points, ex_points).all(axis=1).sum() / len(ex_mito_points)
print(f"Overlap of mito and neuron meshes: {ex_mito_mask}")

# Convert mesh coordinates from nm -> mip0 voxel coordinates (8 nm)
ex_points_px = np.floor(ex_points / np.array([8, 8, 8], dtype=float)).astype(int)
ex_mito_points_px = np.floor(ex_mito_points / np.array([8, 8, 8], dtype=float)).astype(int)

# Convert mip0 coords to image-mip coords with per-axis scale factors.
# This avoids treating z as downsampled when it often is not.
base_res = np.array(vol.scales[0]['resolution'][:3], dtype=float)
curr_res = np.array(vol.scale['resolution'][:3], dtype=float)
mip_factor_xyz = curr_res / base_res
print(f"mip factors xyz @ mip{mip}: {mip_factor_xyz.tolist()}")

ex_points_img = np.floor(ex_points_px / mip_factor_xyz).astype(int)
ex_mito_points_img = np.floor(ex_mito_points_px / mip_factor_xyz).astype(int)

# get minimal bounding box of neuron mesh
# x_min, y_min, z_min = np.min(ex_points_px, axis=0)
# x_max, y_max, z_max = np.max(ex_points_px, axis=0)

#  this obviously just crashes lol 
# just get an example z frame 
# pick the z slice (in current image mip coordinates) with most vertices
z_vals, z_counts = np.unique(ex_points_img[:, 2], return_counts=True)
random_z_img = int(z_vals[np.argmax(z_counts)])
print(
    f"chosen z_img={random_z_img}  ({z_counts.max()} neuron vertices, "
    f"{(ex_mito_points_img[:, 2] == random_z_img).sum()} mito vertices)"
)

ex_points_img_z = ex_points_img[ex_points_img[:, 2] == random_z_img]
x_min, x_max = np.min(ex_points_img_z[:, 0]), np.max(ex_points_img_z[:, 0])
y_min, y_max = np.min(ex_points_img_z[:, 1]), np.max(ex_points_img_z[:, 1])

#%%
x_min_img = int(x_min)
x_max_img = int(x_max) + 1  # python slices are exclusive on the upper bound
y_min_img = int(y_min)
y_max_img = int(y_max) + 1

vol_neu = vol[x_min_img:x_max_img, y_min_img:y_max_img, random_z_img, 0]
print(f"Loaded array shape: {vol_neu.shape}")
#%%


img = np.squeeze(vol_neu).T   # display as (y, x) for imshow


def points_to_mask(pts_img, z_frame_img, x0_img, y0_img, shape):
    """Bool mask (same shape as img) with True at each vertex in this z-slice."""
    in_z = pts_img[:, 2] == int(z_frame_img)
    xy = pts_img[in_z, :2]
    lx = xy[:, 0] - int(x0_img)  # image-local x
    ly = xy[:, 1] - int(y0_img)  # image-local y
    valid = (lx >= 0) & (lx < shape[1]) & (ly >= 0) & (ly < shape[0])
    mask = np.zeros(shape, dtype=bool)
    mask[ly[valid], lx[valid]] = True
    return mask

frame_neuron = points_to_mask(ex_points_img, random_z_img, x_min_img, y_min_img, img.shape)
frame_mito = points_to_mask(ex_mito_points_img, random_z_img, x_min_img, y_min_img, img.shape)

raw_neuron_px = int(frame_neuron.sum())
raw_mito_px = int(frame_mito.sum())
coverage = frame_neuron.mean() * 100
print(f"neuron mask: {raw_neuron_px} px  |  mito mask: {raw_mito_px} px  |  neuron coverage: {coverage:.6f}%")

if VIS_DILATION_RADIUS > 0:
    footprint = morphology.disk(VIS_DILATION_RADIUS)
    frame_neuron = morphology.dilation(frame_neuron, footprint)
    frame_mito = morphology.dilation(frame_mito, footprint)
    print(
        f"after dilation (r={VIS_DILATION_RADIUS}): "
        f"neuron={int(frame_neuron.sum())} px  mito={int(frame_mito.sum())} px"
    )
further_restrict = [2000,3600, 3500, 4600]
fig, ax = plt.subplots(figsize=(10, 10))
vmin, vmax = np.percentile(img, (0.5, 99.5))
ax.imshow(img[further_restrict[2]:further_restrict[3], further_restrict[0]:further_restrict[1]], cmap='gray', vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')

# RGBA overlays — red = neuron, blue = mito
for mask, colour in [(frame_neuron, [1, 0.2, 0.2, 0.2]),
                     (frame_mito,   [0.2, 0.4, 1.0, 0.2])]:
    overlay = np.zeros((*img.shape, 4), dtype=float)
    overlay[mask] = colour
    ax.imshow(overlay[further_restrict[2]:further_restrict[3], further_restrict[0]:further_restrict[1]], aspect='auto', interpolation='nearest', resample=False)

    # Make sparse masks visible even when the image is highly downsampled onscreen.
    yy, xx = np.nonzero(mask[further_restrict[2]:further_restrict[3], further_restrict[0]:further_restrict[1]])
    if yy.size:
        ax.scatter(xx, yy, s=4, c=[colour[:3]], alpha=min(1.0, colour[3] + 0.2),
                   marker='s', linewidths=0)

plt.show()

# %%
cutout = vol_neu.squeeze()   # → (x, y, z) = (300, 300, 10)
