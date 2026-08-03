from cloudvolume import CloudVolume, Bbox
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, segmentation, feature, morphology
import matplotlib.patches as mpatches
import fafbseg
from fafbseg import xform, flywire
from pathlib import Path
from PIL import Image


em_path = 'precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig'
em_clahe_path = 'precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe'

seg_path = 'precomputed://gs://fafb-ffn1-20200412/segmentation'

# first get segmentation of example neuron
seg_id = 710435991
mip = 0
# VIS_DILATION_RADIUS = 2  # set to 0 to keep only exact mesh-vertex pixels
cv_seg = CloudVolume(seg_path, use_https=True, fill_missing=True)
vol = CloudVolume(
    em_clahe_path,
    use_https=True,
    mip=mip,
    progress=True,
    fill_missing=False,
)

#%% FFN1 mesh of the neuron of interest (vertices in FAFB14 nm)
mesh = cv_seg.mesh.get(seg_id)[seg_id]
vertices_nm = mesh.vertices

# Use an actual mesh vertex, not component-wise minima.
# A vertex near the centre of the mesh is less likely to lie at an extreme.
centre_nm = np.median(vertices_nm, axis=0)

# Convert physical nm coordinates to voxel coordinates at vol.mip
resolution_nm = np.asarray(vol.resolution, dtype=float)
centre_vox = np.floor(centre_nm / resolution_nm).astype(np.int64)

print("EM resolution:", vol.resolution)
print("EM bounds:", vol.bounds)
print("Centre nm:", centre_nm)
print("Centre vox:", centre_vox)
print("Inside volume:", vol.bounds.contains(centre_vox))

# to examine in neuroglancer
def show_neuroglancer_coordinates(vertices_nm):
    ''' This is to convert nanometre coordinates to flywire neuroglancer coordinates to view online'''
    centre_fafb14_nm = np.median(vertices_nm, axis=0).reshape(1, 3)

    centre_flywire_nm = xform.fafb14_to_flywire(
        centre_fafb14_nm,
        coordinates="nm",
    )[0]

    centre_flywire_vox = centre_flywire_nm / np.array([4, 4, 40])

    print("FlyWire Neuroglancer XYZ:", centre_flywire_vox)
    print(
        "Pasteable:",
        ",".join(map(str, np.round(centre_flywire_vox).astype(int))),
    )
    return centre_flywire_vox

show_neuroglancer_coordinates(vertices_nm)

x, y, z = centre_vox
y+=100

half_width = 80

x0 = int(x - half_width)
x1 = int(x + half_width)
y0 = int(y - half_width)
y1 = int(y + half_width)
z = int(z)

z_range = 6

bbox = Bbox(
    (x0, y0, z),
    (x1, y1, z + z_range),
)

# Ensure the request does not extend outside the volume
bbox = Bbox.intersection(bbox, vol.bounds)

cutout = np.asarray(vol.download(bbox))

print("Requested bbox:", bbox)
print("Cutout shape:", cutout.shape)
print("Cutout dtype:", cutout.dtype)
print("Cutout range:", cutout.min(), cutout.max())
print("Nonzero pixels:", np.count_nonzero(cutout))

image = np.squeeze(cutout)
if image.max() > image.min():
    vmin, vmax = np.percentile(image, (0.5, 99.5))
else:
    vmin, vmax = None, None

# fig, ax = plt.subplots(1,1, figsize=(5,5))
# ax.imshow(image[...,0].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
# ax.set_title(f"FAFB14 EM, z={z}")


fig, ax = plt.subplots(3, 2, figsize=(10, 15))
ax = ax.flatten()

for e, z_val in enumerate(np.arange(z, z+6)):
    img = image[..., e]
    ax[e].imshow(img.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    ax[e].set_title(f"FAFB14 EM, z={z_val}")
#%%
def to_uint8(plane, vmin, vmax):
    """Linear contrast stretch of one plane to 0-255, matching imshow(vmin, vmax)."""
    plane = np.asarray(plane, dtype=np.float32)
    if vmax is None or vmin is None or vmax <= vmin:
        return np.zeros(plane.shape, dtype=np.uint8)
    scaled = (plane - vmin) / (vmax - vmin)
    return (np.clip(scaled, 0, 1) * 255).round().astype(np.uint8)


def save_cutouts(image, output_folder, name, vmin=None, vmax=None, scale=1):
    """Save each z plane of `image` (x, y, z) as a lossless grayscale PNG.

    One PNG pixel = one EM voxel, so annotation coordinates map straight back to
    voxels. `scale` upsamples by an integer factor with nearest-neighbour, which
    keeps that mapping exact (voxel = pixel // scale) while making the image
    easier to annotate by eye. Contrast defaults to the 0.5-99.5 percentiles of
    the whole stack, so all planes share one mapping.
    """
    assert (len(image.shape) <= 3), print(f'image has wrong shape: {image.shape}')
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(image, (0.5, 99.5))

    n_z = image.shape[2]
    for zz in range(n_z):
        # .T + origin='lower' in imshow == transpose then flip rows for a PNG,
        # whose first row is the top one.
        plane = np.flipud(image[..., zz].T)
        img = Image.fromarray(to_uint8(plane, vmin, vmax), mode="L")
        if scale != 1:
            img = img.resize((img.width * scale, img.height * scale),
                             resample=Image.NEAREST)
        img.save(output_folder / f'ex_{name}_{zz}.png', optimize=True)
    print(f'saved {n_z} planes of {img.size[0]}x{img.size[1]} px to {output_folder}')

save_cutouts(image, 'data/fafb_em/', name='114927-58280-1004', scale=1)

#%%

vol = cv_seg[x0:x1, y0:y1, z]

def remap_seg(seg, b=8, seed=23):    
    u_ids = np.unique(seg)
    np.random.seed(seed)
    np.random.shuffle(u_ids)
    mapping = np.vectorize(dict(zip(u_ids, np.random.randint(0, 2**b-1, size=len(u_ids)))).get)
    
    remapped_seg = mapping(seg).astype(int)
    
    return remapped_seg

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(remap_seg(vol[..., 0, 0]), cmap="nipy_spectral")

