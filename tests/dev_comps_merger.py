#%%
import numpy as np
import pandas as pd
from typing import Literal

from scipy.spatial.distance import cdist
from pymatgen.core import Composition

from hsdu.dataset import Dataset, D2TableDataset
from hsdu.utils.conversion_utils import almost_equals_pymatgen_atomic_fraction, element_list_iupac_ordered
from hsdu.utils.duplicate import make_duplicates_group, distance_matrix, compare_dupl_groups, compare_dupl_groups_old2new


                    
hsdset_path=r"sgfhsgfh"
hsd = Dataset(hsdset_path)

onehot_fracs = hsd.onehot_fracs()
for idx, row in enumerate(hsd):
    a= hsd.idx2aux["comps_pymatgen"][idx]
    onehot_frac = hsd.onehot_codec.encode(a)
    d=hsd.onehot_codec.decode(onehot_frac)
    assert almost_equals_pymatgen_atomic_fraction(a, d, rtol=1e-9)
    assert np.abs(sum(onehot_frac)-1)<1e-9 # check, sum(onehot_frac) ~ 1


# # merge entries with close compositions
assert hsd.index==list(range(len(hsd)))
elements_sets = [hsd[i]["elements_set"] for i in hsd.index]
print(pd.Series(elements_sets).value_counts().head(10))

# ## make distance matrix
# chebyshev distance
linfty_dist_matrix = distance_matrix(onehot_fracs, onehot_fracs, metric="l_infty", elemental_set=[row["elements_set"] for row in hsd])
# L1 distance
l1_dist_matrix = distance_matrix(onehot_fracs, onehot_fracs, metric="l1", elemental_set=[row["elements_set"] for row in hsd])

dup_group, group_rows = make_duplicates_group(hsd._df.index, 
                                              l1_dist_matrix, linfty_dist_matrix, 
                                              elements_sets,
                                              linfty_cutoff=0.01, l1_cutoff=0.02)



print(len(dup_group))
print("dinfty_cutoff=0.01, d1_cutoff=0.02")
print("elements set of 10 largest groups:")
for i in pd.Series(group_rows).value_counts().index[:10]:
    print(f"{hsd._df.loc[i, 'elements_set']} shown {len(dup_group[i])} times. indices: {dup_group[i]}")

# prepare old 'merge_duplicates' attributes
hsd.pymatgen_duplicates(rtol=0.02)
print(len(hsd)-len(hsd.duplicated_comps)+len(hsd.duplicated_comps_group))#should be 361

# prepare list of group's compositions (list[list[Composition]])
group_comps = []
for dup_idx_set in dup_group.values():
    group_comps.append([hsd[dup_idx]['comps_pymatgen'] for dup_idx in dup_idx_set])


#%% it takes long time!!
# group distances; inbetween new groups

new_groups_distance_matrix=compare_dupl_groups(group_comps, dup_group.keys())
for dist_metric, dist_df in new_groups_distance_matrix.items():
    min_dist = np.nanmin(dist_df)
    mask=(dist_df == min_dist)
    print(f"min_{dist_metric}: {min_dist}")
    other_metric = 'l1' if dist_metric=='linfty' else 'linfty'
    group0_idx, group1_idx = dist_df.index[mask.any()]
    other_metric_val = compare_dupl_groups([[hsd[i]['comps_pymatgen'] for i in dup_group[group0_idx]],[hsd[i]['comps_pymatgen'] for i in dup_group[group1_idx]]],
                                           group_names=[group0_idx, group1_idx], print_output=False)
    other_metric_val_np = other_metric_val[other_metric].to_numpy().flatten()
    other_metric_val = set(other_metric_val_np[~np.isnan(other_metric_val_np)])
    assert len(other_metric_val)==1
    print(f"where {other_metric}: {other_metric_val}")
    print(f"group 0: {[hsd[idx]['comps_pymatgen'] for idx in dup_group[group0_idx]]}")
    print(f"group 1: {[hsd[idx]['comps_pymatgen'] for idx in dup_group[group1_idx]]}")
    
#%%  it takes long time!!
# bound to elements group;
new_groups_distance_matrix=compare_dupl_groups(group_comps, dup_group.keys(), ignore_cross_elemental_set=True)
for dist_metric, dist_df in new_groups_distance_matrix.items():
    min_dist = np.nanmin(dist_df)
    mask=(dist_df == min_dist)
    print(f"min_{dist_metric}: {min_dist}")
    other_metric = 'l1' if dist_metric=='linfty' else 'linfty'
    group0_idx, group1_idx = dist_df.index[mask.any()]
    other_metric_val = compare_dupl_groups([[hsd[i]['comps_pymatgen'] for i in dup_group[group0_idx]],[hsd[i]['comps_pymatgen'] for i in dup_group[group1_idx]]],
                                           group_names=[group0_idx, group1_idx], print_output=False)
    other_metric_val_np = other_metric_val[other_metric].to_numpy().flatten()
    other_metric_val = set(other_metric_val_np[~np.isnan(other_metric_val_np)])
    assert len(other_metric_val)==1
    print(f"where {other_metric}: {other_metric_val}")
    print(f"group 0: {[hsd[idx]['comps_pymatgen'] for idx in dup_group[group0_idx]]}")
    print(f"group 1: {[hsd[idx]['comps_pymatgen'] for idx in dup_group[group1_idx]]}")
    
#%%
# group distance; inbetween old groups
hsd.pymatgen_duplicates(rtol=0.02)
old_group_comps = [[hsd[i]['comps_pymatgen'] for i in v.keys()] for k, v in hsd.duplicated_comps_group.items()]

new_groups_distance_matrix=compare_dupl_groups(old_group_comps, hsd.duplicated_comps_group.keys(), ignore_cross_elemental_set=True)
for dist_metric, dist_df in new_groups_distance_matrix.items():
    min_dist = np.nanmin(dist_df)
    mask=(dist_df == min_dist)
    print(f"min_{dist_metric}: {min_dist}")
    other_metric = 'l1' if dist_metric=='linfty' else 'linfty'
    group0_idx, group1_idx = dist_df.index[mask.any()]
    other_metric_val = compare_dupl_groups([[hsd[i]['comps_pymatgen'] for i in hsd.duplicated_comps_group[group0_idx].keys()],[hsd[i]['comps_pymatgen'] for i in hsd.duplicated_comps_group[group1_idx].keys()]],
                                           group_names=[group0_idx, group1_idx], print_output=False)
    other_metric_val_np = other_metric_val[other_metric].to_numpy().flatten()
    other_metric_val_out = set(other_metric_val_np[~np.isnan(other_metric_val_np)])
    assert len(other_metric_val_out)==1
    print(f"where {other_metric}: {other_metric_val_out.pop()}")
    print(f"group {group0_idx}: {[hsd[idx]['comps_pymatgen'] for idx in hsd.duplicated_comps_group[group0_idx]]}")
    print(f"group {group1_idx}: {[hsd[idx]['comps_pymatgen'] for idx in hsd.duplicated_comps_group[group1_idx]]}")
#%% 
# test param, dupl_group - to prevent data leakage (inbetween train-test split: dinfty_cutoff=0.05, d1_cutoff=0.12) 
linfty_cutoff=0.05
l1_cutoff=0.12

dup_group, group_rows = make_duplicates_group(hsd._df.index,
                                              l1_dist_matrix, linfty_dist_matrix,
                                              elements_sets,
                                              linfty_cutoff=0.05, l1_cutoff=0.12,
                                              cross_elements_set=True)

print(len(dup_group))
print("elements set of 10 largest group:")
for i in pd.Series(group_rows).value_counts().index[:10]:
    print(f"{hsd._df.loc[i, 'elements_set']} shown {len(dup_group[i])} times. indices: {dup_group[i]}")

# %% 
# #### compare with old method - BaseDataset.pymatgen_duplicates() ####

linfty_cutoff=0.02
l1_cutoff=0.02
dup_group, group_rows = make_duplicates_group(hsd._df.index, 
                                              l1_dist_matrix, linfty_dist_matrix, 
                                              elements_sets,
                                              linfty_cutoff=linfty_cutoff, l1_cutoff=l1_cutoff, cross_elements_set=False)
print(len(dup_group))

# pymatgen_duplicates():
#   - initializes `self.duplicated_comps_group=dict()` and `self.duplicated_comps=set()`.
#   - `self.duplicated_comps_group.keys()` are entry-index of group's first item.
hsd.pymatgen_duplicates() 
old2new = compare_dupl_groups_old2new(hsd,dup_group, group_rows)

# %% 
# #### Compare with old method - reading old json file #####
old_log = pd.read_json("hsf")
log_duplicated_group = old_log["duplicated_comps"].dropna().to_dict()

print(len(log_duplicated_group.keys()))
log_duplicated_group = {int(k):{int(gk):gv for gk, gv in v.items()} for k, v in log_duplicated_group.items()}

log_duplicated_comps = set()
for k, v in log_duplicated_group.items():
    log_duplicated_comps.update(v.keys())

assert log_duplicated_comps==hsd.duplicated_comps
assert log_duplicated_group == hsd.duplicated_comps_group

