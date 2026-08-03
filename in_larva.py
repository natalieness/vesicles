import pymaid
from pymaid import tiles

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pymaid_creds import url, name, password, token
rm = pymaid.CatmaidInstance(url, token, name, password)


#%%
# in nm 
xy_dist = 3000
z_dist = 500
x0, y0, z0 = 67000, 78000, 141650
x1, y1, z1 = x0 + xy_dist, y0 + xy_dist, z0 + z_dist

bbox = [x0, x1, y0, y1, z0, z1]

print(bbox)

job = tiles.TileLoader(bbox, stack_id=1, coords='NM')
job.load_in_memory()

arr = job.img
print(f"Loaded array shape: {arr.shape}")
# %%
n_z = 2
fig, ax = plt.subplots(1,n_z, figsize=(30, 30))
if n_z >1:
    ax = ax.flatten()
for z in range(0,n_z):
    vmin, vmax = np.percentile(arr[..., z], (0., 100. ))
    ax[z].imshow(arr[...,  z], cmap='gray', vmin=vmin, vmax=vmax)
    ax[z].axis('off')
fig.tight_layout()
# %%
