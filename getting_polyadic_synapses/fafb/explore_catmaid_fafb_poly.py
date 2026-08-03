
import pymaid
import pandas as pd
import numpy as np

#%%

rm = pymaid.connect_catmaid(
    server="https://fafb.catmaid.virtualflybrain.org/",
    api_token=None,
)

skeleton_ids = pymaid.get_skeleton_ids(remote_instance=rm)
names = pymaid.get_names(skeleton_ids, remote_instance=rm)
names = {int(k): v for k, v in names.items()}

# restrict to just some skeleton ids because otherwise cant download them in one go 
skeleton_ids_subset = list(skeleton_ids)[:50]
connectors = pymaid.get_connectors(skeleton_ids_subset, remote_instance=rm)

presyn_connectors = connectors[connectors['type'] == 'Presynaptic']['connector_id'].unique().tolist()
print(len(presyn_connectors), 'presynaptic connectors')

connector_details = pymaid.get_connector_details(
    presyn_connectors, remote_instance=rm
)

#%%
connector_details['pre_name'] = connector_details['presynaptic_to'].map(names)
connector_details['n_post'] = connector_details['postsynaptic_to'].apply(len)

connector_details.groupby('pre_name')['n_post'].mean().hist(bins=50, grid=False)

connector_details['MBON'] = connector_details['pre_name'].str.contains('MBON')
connector_details['KC'] = connector_details['pre_name'].str.contains('KC')

print(connector_details.groupby('MBON')['n_post'].mean())
print(connector_details.groupby('KC')['n_post'].mean())

# %%
