# hemibrain dataset from neuroglancer 

from cloudvolume import CloudVolume
import numpy as np
import matplotlib.pyplot as plt


em_aligned = 'n5://gs://flyem_cns_z0720_07m_dvidcoords_n5/' # em path not correct, also would need to open with tensorstore 
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
xy_space = 300
x0= 19500
y0 = 21610
x1, y1 = x0 + xy_space, y0 + xy_space
z = 18141
cutout = vol[x0:x1, y0:y1, z, 0]
    
fig, ax = plt.subplots(figsize=(10, 10))
vmin, vmax = np.percentile(cutout, (0.5, 99.5))
ax.imshow(cutout[..., 0], cmap='gray', vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(2, 3, figsize=(30, 20))
ax = ax.flatten()
for z in np.arange(18140, 18146):
    cutout = vol[x0:x1, y0:y1, z, 0]
    vmin, vmax = np.percentile(cutout, (0.5, 99.5))
    ax[z-18140].imshow(cutout[..., 0], cmap='gray', vmin=vmin, vmax=vmax)
    ax[z-18140].set_title(f'z={z}')

#%% segmentation and mesh 

cv_seg = CloudVolume(seg_path, use_https=True, fill_missing=True)

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
