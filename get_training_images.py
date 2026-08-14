''' To download FAFB EM image cutouts for ground truth annotation etc. 
Idea is to start off with a couple of neurons of interest, find 3D volumes with some presynaptic terminals, 
and then save those volumes as individual z frames for annotation.
'''

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
import navis


em_path = 'precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig'
em_clahe_path = 'precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe'
seg_path = 'precomputed://gs://fafb-ffn1-20200412/segmentation'


#%% get 
# first get segmentation of example neuron

url = (
    "https://raw.githubusercontent.com/"
    "flyconnectome/flywire_annotations/main/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)

ann = pd.read_csv(url, sep="\t")

kcs = ann[ann["cell_class"] == "Kenyon_Cell"][['root_id', 'top_nt', 'side', 'super_class', 'cell_class', 'cell_sub_class', 'supertype', 'cell_type', 'status']]
dans = ann[ann["cell_class"] == "DAN"][['root_id', 'top_nt', 'side', 'super_class', 'cell_class', 'cell_sub_class', 'supertype', 'cell_type', 'status']]

n_per_class = 15

kc_roots = kcs.sample(n=n_per_class, random_state=23)['root_id'].values
dan_roots = dans.sample(n=n_per_class, random_state=23)['root_id'].values

#%% functions here 
def get_presynaptic_centers(root_id, n_areas_per_neuron=3):
    syn = flywire.get_synapses(
        root_id,
        pre=True,
        post=False,
        attach=False,
        materialization=783,
        filtered=True,
        dataset="public"
    )

    # find presynapses with multiple on the same z plane 
    zs = syn.groupby('pre_z')['pre_z'].count().sort_values(ascending=False).index[:n_areas_per_neuron]
    coords = []
    for z in zs:
        coords.append(syn[syn['pre_z'] == z][['pre_x', 'pre_y', 'pre_z']].iloc[0,:].to_numpy())

    centers_fafb14 = navis.xform_brain(
        coords[0].reshape(1, 3),
        source="FLYWIRE",
        target="FAFB14"
    )
    return centers_fafb14

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

def view_example(root_id, centers_fafb14):
    # view here
    kc_meshes = flywire.get_mesh_neuron(
        root_id,
        dataset="public"
    )

    navis.plot3d(kc_meshes)

    vox_xyz = show_neuroglancer_coordinates(centers_fafb14[0].reshape(1, 3))

    rid = flywire.locs_to_segments(
        vox_xyz.reshape(1, 3),
        coordinates="voxel",
        dataset="public",
        timestamp="mat_783"
    )

    print(rid)

    url_viw = flywire.encode_url(
        segments=[rid],
        dataset="public",
        layout="3d",
        open=True
    )
    return url_viw

def load_em_data(em_path=em_clahe_path, mip=0):
    vol = CloudVolume(
        em_clahe_path,
        use_https=True,
        mip=mip,
        progress=True,
        fill_missing=False,
    )
    return vol

def get_em_cutout(centers_fafb14, vol, half_width=80, z_range=6):
    res = vol.resolution
    center_vox = np.round(centers_fafb14[0] / res).astype(int)

    x, y, z = center_vox

    x0 = int(x - half_width)
    x1 = int(x + half_width)
    y0 = int(y - half_width)
    y1 = int(y + half_width)
    z = int(z)

    bbox = Bbox(
        (x0, y0, z),
        (x1, y1, z + z_range),
    )

    # Ensure the request does not extend outside the volume
    bbox = Bbox.intersection(bbox, vol.bounds)

    cutout = np.asarray(vol.download(bbox))
    image = np.squeeze(cutout)
    image_name = f'x{x0}-{x1}_y{y0}-{y1}_z{z}-{z+z_range}'
    return image, image_name

def plot_image(image):
    z_rang = image.shape[2]

    if image.max() > image.min():
        vmin, vmax = np.percentile(image, (0.5, 99.5))
    else:
        vmin, vmax = None, None

    fig, ax = plt.subplots(z_rang//2, 2, figsize=(10, 15))
    ax = ax.flatten()

    for e, z_val in enumerate(np.arange(0, z_rang, 1)):
        img = image[..., e]
        ax[e].imshow(img.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax[e].set_title(f"FAFB14 EM, z={z_val}")

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

#%% get presynaptic terminal locations for each 

n_areas_per_neuron = 3
half_width = 80
z_range = 6
PLOT_EXAMPLES = False
data_to = 'data/fafbv783_em/'

# load em data once 
vol = load_em_data(em_clahe_path, mip=0)

# iterate here later 
root_i = kc_roots[0]
root_names = ['KC']*len(kc_roots) + ['DAN']*len(dan_roots)
for root_i, root_nam in zip(list(kc_roots) + list(dan_roots), root_names):
    print(root_i, root_nam)
    centers_fafb14 = get_presynaptic_centers(root_i, n_areas_per_neuron=n_areas_per_neuron)

    # view_example(root_i, centers_fafb14)

    # get em image 
    image, image_name = get_em_cutout(centers_fafb14, vol, half_width=half_width, z_range=z_range)
    if PLOT_EXAMPLES:
        plot_image(image)
        
    image_name = f'{root_nam}_{root_i}_{image_name}'
    save_cutouts(image, data_to, name=image_name, scale=1)

# %%
