
''' Visualise polyadic contact point in 3d mesh '''

from cloudvolume import CloudVolume, from_cloudpath, Bbox
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.draw import polygon as sk_polygon
import matplotlib.patches as mpatches
import trimesh

em_clahe_path = 'precomputed://gs://flyem-male-cns/em/em-clahe-jpeg'
seg_path      = 'precomputed://gs://flyem-male-cns/v0.9/segmentation'
presyn_path = 'precomputed://gs://flyem-male-cns/v0.9/male-cns-v0.9-synapses-precomputed'
postsyn_path = 'precomputed://gs://flyem-male-cns/v0.9/male-cns-v0.9-synapses-precomputed'

#%% load clouvolume objects 
mip=0
vol = CloudVolume(em_clahe_path, use_https=True, mip=mip, progress=True, fill_missing=False)
cv_seg = CloudVolume(seg_path, use_https=True, fill_missing=True)


#%% get specific presynaptic site of interest 
# just choose for this purpose

pre_x, pre_y, pre_z = 50415, 17152, 17674
pre_neu_id = 161079

post_ids = [54647, 43040, 711545612]

pre_mesh = cv_seg.mesh.get(pre_neu_id)[pre_neu_id]
post_meshes = [cv_seg.mesh.get(post_id)[post_id] for post_id in post_ids]

# quick show em around point 
xy_dist = 100
img = vol[(pre_x-xy_dist)//2**mip: (pre_x+xy_dist)// 2**mip, (pre_y-xy_dist)//2**mip: (pre_y+xy_dist)//2**mip, pre_z//2**mip, 0].squeeze()
vmin, vmax = np.percentile(img, (0.5, 99.5))
plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')


#%% ── build trimesh objects from the 3-D meshes ────────────────────────────────
# Build once, query many times.  process=False preserves the original geometry.
print("Building trimesh objects …")
tm_pre = trimesh.Trimesh(
    vertices=np.asarray(pre_mesh.vertices, dtype=float),
    faces=np.asarray(pre_mesh.faces),
    process=False,
)
tm_posts = []
for post_mesh in post_meshes:
    tm_post = trimesh.Trimesh(
        vertices=np.asarray(post_mesh.vertices, dtype=float),
        faces=np.asarray(post_mesh.faces),
        process=False,
    )
    tm_posts.append(tm_post)

#%% 
def mip_point(point, mip=mip):
    return point // 2**mip
xyz_dist = 100
img_bbox = Bbox(mip_point(np.array([pre_x-xyz_dist, pre_y-xyz_dist, pre_z-xyz_dist])),
                mip_point(np.array([pre_x+xyz_dist, pre_y+xyz_dist, pre_z+xyz_dist])))

