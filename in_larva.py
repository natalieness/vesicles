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
fig, ax = plt.subplots(figsize=(10, 10))
vmin, vmax = np.percentile(arr[..., 0], (0., 100. ))
ax.imshow(arr[...,  0], cmap='gray', vmin=vmin, vmax=vmax)
# %%
