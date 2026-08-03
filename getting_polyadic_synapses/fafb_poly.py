''' This script is to estimate which synapses are clustered on the same polyadic synapse in the FAFB dataset.'''


from cloudvolume import CloudVolume
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, segmentation, feature, morphology
import matplotlib.patches as mpatches
import fafbseg
from fafbseg import xform, flywire
from pathlib import Path


em_path = 'precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig'
em_clahe_path = 'precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe'

seg_path = 'precomputed://gs://fafb-ffn1-20200412/segmentation'

# The Buhmann synapses are a neuroglancer annotation layer, not an image volume, so loading
# them with CloudVolume fails ("KeyError: 'scales'"). Use the Princeton v783 synapse table
# instead - the same file explore_fafb_poly.py uses.
try:
    HERE = Path(__file__).resolve().parent
except NameError:               # running as interactive #%% cells
    HERE = Path.cwd()
syn_table_path = HERE / 'fafb' / 'fafb_data' / 'fafb_v783_princeton_synapse_table.csv.gz'

# v783 FlyWire root ids in the table share this leading prefix; the columns store the remainder
FLYWIRE_ID_PREFIX = 720575940


#%% get area of interest

# first get segmentation of example neuron
seg_id = 710435991
mip = 1
# VIS_DILATION_RADIUS = 2  # set to 0 to keep only exact mesh-vertex pixels
cv_seg = CloudVolume(seg_path, use_https=True, fill_missing=True)
vol = CloudVolume(
    em_clahe_path,
    use_https=True,
    mip=mip,
    progress=True,
    fill_missing=False,
)

# FFN1 mesh of the neuron of interest (vertices in FAFB14 nm)
ex = cv_seg.mesh.get(seg_id)[seg_id]
ex_points = ex.vertices


#%% map FFN1 segment -> FlyWire v783 root id (inverse of flywire_root_to_ffn1 in explore_fafb_poly.py)

def ffn1_to_flywire_root(ffn1_verts, n_points=1000, flywire_dataset='public', seed=1, return_all=False):
    """Map an FFN1 segment to the best-matching FlyWire v783 root id via spatial overlap.

    Samples the FFN1 mesh vertices, transforms FAFB14 nm -> FlyWire nm, looks up the FlyWire
    segment id at each point, and returns the root id covering the most sampled points.
    'public' is the FlyWire v783 release, so returned roots line up with the Princeton table.
    """
    verts = np.asarray(ffn1_verts, dtype=float)
    if verts.size == 0:
        raise ValueError('No FFN1 mesh vertices provided')

    rng = np.random.default_rng(seed)
    if len(verts) > n_points:
        verts = verts[rng.choice(len(verts), size=n_points, replace=False)]

    flywire_nm = xform.fafb14_to_flywire(verts, coordinates='nm')
    roots = flywire.locs_to_segments(flywire_nm, coordinates='nm', dataset=flywire_dataset)

    s = pd.Series(roots, name='flywire_root_id')
    s = s[s.ne(0)]              # 0 == background / failed lookup
    if s.empty:
        raise ValueError(f'No non-zero FlyWire roots found for FFN1 seg {seg_id}')

    out = (s.value_counts()
             .rename_axis('flywire_root_id')
             .reset_index(name='n_points'))
    out['frac'] = out['n_points'] / out['n_points'].sum()
    if return_all:
        return out

    # Pull each field from its column: FlyWire root ids exceed 2**53, so reading them
    # via out.iloc[0] (a mixed int/float row) would silently corrupt them to float64.
    best_frac = float(out['frac'].iloc[0])
    best = {
        'flywire_root_id': int(out['flywire_root_id'].iloc[0]),
        'n_points': int(out['n_points'].iloc[0]),
        'frac': best_frac,
        'ambiguous': bool(best_frac < 0.5),
    }
    return best


match = ffn1_to_flywire_root(ex_points)
flywire_root = int(match['flywire_root_id'])
root_short = flywire_root % 1_000_000_000   # strip the 720575940 prefix to match table columns
print(f"FFN1 {seg_id} -> FlyWire root {flywire_root} "
      f"(frac={match['frac']:.2f}, ambiguous={match['ambiguous']})")


#%% pull this neuron's synapse locations from the Princeton v783 table

syn = pd.read_csv(syn_table_path, compression='gzip')
syn.rename(columns={'pre_root_id_720575940': 'pre', 'post_root_id_720575940': 'post'}, inplace=True)

pre_syn = syn[syn['pre'] == root_short]     # neuron is presynaptic  -> its outputs
post_syn = syn[syn['post'] == root_short]   # neuron is postsynaptic -> its inputs
print(f"FFN1 {seg_id}: {len(pre_syn)} output and {len(post_syn)} input synapses")

# synapse locations in FlyWire nm (the table's native space)
pre_syn_points_flywire = pre_syn[['pre_x', 'pre_y', 'pre_z']].to_numpy(dtype=float)
post_syn_points_flywire = post_syn[['post_x', 'post_y', 'post_z']].to_numpy(dtype=float)

# same locations transformed to FAFB14 nm so they line up with the FFN1 mesh (ex_points)
ex_syn_points = (xform.flywire_to_fafb14(pre_syn_points_flywire, coordinates='nm')
                 if len(pre_syn_points_flywire) else pre_syn_points_flywire)
ex_syn_post_points = (xform.flywire_to_fafb14(post_syn_points_flywire, coordinates='nm')
                      if len(post_syn_points_flywire) else post_syn_points_flywire)

