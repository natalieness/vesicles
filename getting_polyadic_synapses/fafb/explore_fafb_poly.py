### Note newer princeton tables don't have unique identifiers for the connectors - just pre and post xyz positions which are not exactly overlapping. 

### older data from v783 release does have them - but should check how much has changed since. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns




### tried older buhmann data for polyadic synapses but connector ids dont make sense so ignore
# full old only available straight from gs 
#gs_address = 'gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/fafb_783/fafb_783_synapses.parquet'
# downloaded locally for speed 
# old_path = 'adult/fafb/fafb_data/fafb_783_synapses.parquet'

# # v783 release from Buhmann et al and Heinrich et al. 
# syn = pd.read_parquet(old_path)
 
# print(f'Number of individual synapse connections: new  {syn_new.shape[0]}')
# print(f'Number of individual synapse connections: old  {syn.shape[0]}')

# # create a con table for the older data - note to make it more efficient remove any cols that could be lookup tables e.g. syn prediction 
# # note some metadata is pretty cool and should probably be looked at later e.g. post_label is compartment 

# restricted_syn = syn[['id', 'connector_id', 'pre', 'post']]
# con = restricted_syn.groupby(['pre','connector_id'])['post'].apply(list).reset_index(name='post')
# con['n_post'] = con['post'].apply(len)

#%%
new_path = 'adult/fafb/fafb_data/fafb_v783_princeton_synapse_table.csv.gz'
# newer princeton release 
syn = pd.read_csv(new_path, compression='gzip')

# simplify root id 
syn.rename(columns={'pre_root_id_720575940':'pre', 'post_root_id_720575940':'post'}, inplace=True)

# neuron details 
neus = pd.read_csv('adult/fafb/fafb_data/neurons.csv.gz', compression='gzip')
classification = pd.read_csv('adult/fafb/fafb_data/classification.csv.gz', compression='gzip')


# remove root_id start 
neus['root_id'] = neus['root_id'].astype(str)
classification['root_id'] = classification['root_id'].astype(str)
neus['root_id'] = neus['root_id'].str.replace('720575940', '', regex=False)
classification['root_id'] = classification['root_id'].str.replace('720575940', '', regex=False)
neus['root_id'] = neus['root_id'].astype(int)
classification['root_id'] = classification['root_id'].astype(int)

skid_to_nt = dict(zip(neus['root_id'], neus['nt_type']))
skid_to_group = dict(zip(neus['root_id'], neus['group']))
skid_to_superclass = dict(zip(classification['root_id'], classification['super_class']))
skid_to_class = dict(zip(classification['root_id'], classification['class']))
skid_to_subclass = dict(zip(classification['root_id'], classification['sub_class']))



#%% functions for interacting with fafb instances 

from fafbseg import xform, google

def flywire_pos_to_ffn1(x, y, z):
    flywire_xyz_nm = np.array([x, y, z], dtype=float).reshape(1, 3)
    fafb14_xyz_nm = xform.flywire_to_fafb14(
        flywire_xyz_nm,
        coordinates="nm",
        mip=2
    )
    fafb14_xyz_voxel = [fafb14_xyz_nm[0, 0]//4, fafb14_xyz_nm[0, 1]//4, fafb14_xyz_nm[0, 2]//40]
    return fafb14_xyz_voxel

def flywire_root_to_ffn1(
    root_id,
    n_points=1000,
    flywire_dataset="public",
    flywire_timestamp="mat_783",
    ffn1_dataset="fafb-ffn1-20200412",
    seed=1,
    return_all=False,
):
    """
    Map one FlyWire/Codex root_id to likely FAFB-FFN1 segment IDs by spatial overlap.

    Method:
      1. Fetch FlyWire mesh for `root_id`.
      2. Sample mesh vertices in FlyWire-space nm.
      3. Transform FlyWire coordinates -> FAFB14 coordinates.
      4. Query Google FAFB-FFN1 segmentation at those points.
      5. Return ranked FFN1 segment IDs by number/fraction of sampled points.

    Returns:
      If return_all=False: dict for best FFN1 match.
      If return_all=True: pandas.DataFrame of all non-zero FFN1 matches.

    Notes:
      - This is not guaranteed one-to-one.
      - 0 FFN1 IDs are discarded as failed/background lookups.
      - `flywire_timestamp="mat_783"` is appropriate for Codex/FlyWire v783-style root IDs.
    """
    import numpy as np
    import pandas as pd
    from fafbseg import flywire, google, xform

    root_id = int(root_id)

    mesh = flywire.get_mesh_neuron(
        root_id,
        dataset=flywire_dataset,
        lod=2,
        progress=False,
    )

    verts = np.asarray(mesh.vertices, dtype=float)
    if verts.size == 0:
        raise ValueError(f"No mesh vertices found for FlyWire root_id={root_id}")

    rng = np.random.default_rng(seed)
    if len(verts) > n_points:
        idx = rng.choice(len(verts), size=n_points, replace=False)
        verts = verts[idx]

    # FlyWire nm -> FAFB14 nm
    fafb14_nm = xform.flywire_to_fafb14(
        verts,
        coordinates="nm",
        mip=2,
    )

    ffn1_ids = google.locs_to_segments(
        fafb14_nm,
        dataset=ffn1_dataset,
        coordinates="nm",
        mip=0,
    )

    s = pd.Series(ffn1_ids, name="ffn1_segment_id")
    s = s[s.ne(0)]  # 0 means invalid/background/no segment

    if s.empty:
        raise ValueError(
            f"No non-zero FFN1 IDs found for FlyWire root_id={root_id}. "
            "Try increasing n_points or checking coordinate/version assumptions."
        )

    out = (
        s.value_counts()
        .rename_axis("ffn1_segment_id")
        .reset_index(name="n_points")
    )
    out["flywire_root_id"] = root_id
    out["frac"] = out["n_points"] / out["n_points"].sum()
    out = out[
        ["flywire_root_id", "ffn1_segment_id", "n_points", "frac"]
    ].sort_values("n_points", ascending=False)

    if return_all:
        return out.reset_index(drop=True)

    best = out.iloc[0].to_dict()
    best["ambiguous"] = bool(best["frac"] < 0.5)
    return best

#%%

#% thresholding - how about 250-500nm?? 
syn_thresh = 250
ex_pre = 608304220 # some mbon

ex_syn = syn[(syn['pre'] == ex_pre)]

def get_syn_dists(syn_df, col_prefix='pre', pre_skid=None):
    if pre_skid is not None:
        syn_df = syn_df[syn_df['pre'] == pre_skid]
    
    pos_cols = [f'{col_prefix}_x', f'{col_prefix}_y', f'{col_prefix}_z']
    pos_array = syn_df[pos_cols].to_numpy(dtype=np.float32, copy=False)
    n = pos_array.shape[0]

    if n == 0:
        return np.empty((0, 0), dtype=np.float32)

    # Faster and lower-memory than explicit broadcasting to (n, n, 3).
    sq_norm = np.einsum('ij,ij->i', pos_array, pos_array)
    dists_sq = sq_norm[:, None] + sq_norm[None, :] - 2.0 * (pos_array @ pos_array.T)
    np.maximum(dists_sq, 0, out=dists_sq)
    return np.sqrt(dists_sq, out=dists_sq)

def group_polyads_based_on_dist_threshold(syn_df, col_prefix='pre', pre_skid=None, dist_thresh=250):
    from scipy.spatial import cKDTree
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix

    if pre_skid is not None:
        working = syn_df[syn_df['pre'] == pre_skid]
    else:
        working = syn_df

    pos_cols = [f'{col_prefix}_x', f'{col_prefix}_y', f'{col_prefix}_z']
    pos = working[pos_cols].to_numpy(dtype=np.float32)
    n = pos.shape[0]
    if n == 0:
        return []

    # query_pairs returns only pairs within threshold — avoids the full n×n matrix
    tree = cKDTree(pos)
    pairs = tree.query_pairs(dist_thresh, output_type='ndarray')  # shape (k, 2)

    if pairs.shape[0] > 0:
        rows, cols = pairs[:, 0], pairs[:, 1]
        data = np.ones(len(rows), dtype=np.bool_)
        adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    else:
        adj = csr_matrix((n, n), dtype=np.bool_)

    _, labels = connected_components(adj, directed=False)

    syn_df.loc[working.index, 'connector_id'] = labels

    return syn_df

ex_syn_ = group_polyads_based_on_dist_threshold(ex_syn, col_prefix='pre', pre_skid=ex_pre, dist_thresh=syn_thresh)
print(f'From {ex_syn.shape[0]} synapses, found {ex_syn_["connector_id"].nunique()} polyadic groups for pre neuron {ex_pre} with threshold {syn_thresh}nm.')
print(f'Average number of posts per polyad: {ex_syn_["connector_id"].value_counts().mean():.2f}')
#%% to check transform coords to google segmentation 



# %%
