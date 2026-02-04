#%%

import pandas as pd
from scipy.spatial.distance import cdist
from pymatgen.core import Composition

from hsdu.dataset import Dataset
from hsdu.utils.conversion_utils import almost_equals_pymatgen_atomic_fraction, element_list_iupac_ordered
from hsdu.utils.duplicate import make_duplicates_group, distance_matrix
#%%
hsdset_path=r"some path"
hsd = Dataset(hsdset_path)

onehot_fracs = hsd.onehot_fracs()
for idx, row in enumerate(hsd):
    a= hsd.idx2aux["comps_pymatgen"][idx]
    onehot_frac = hsd.onehot_codec.encode(a)
    d=hsd.onehot_codec.decode(onehot_frac)
    assert almost_equals_pymatgen_atomic_fraction(a, d, rtol=1e-9)

# # merge entries with close compositions
assert hsd._df.index.tolist()==list(range(len(hsd)))
elements_sets = [hsd[i]["elements_set"] for i in hsd.index]
print(pd.Series(elements_sets).value_counts().head(10))

# ## make distance matrix
# chebyshev distance
linfty_dist_matrix = distance_matrix(onehot_fracs, metric="l_infty", elemental_set=[row["elements_set"] for row in hsd])
# L1 distance
l1_dist_matrix = distance_matrix(onehot_fracs, metric="l1", elemental_set=[row["elements_set"] for row in hsd])

dup_group, group_rows = make_duplicates_group(hsd._df.index, 
                                              l1_dist_matrix, linfty_dist_matrix, 
                                              elements_sets,
                                              linfty_cutoff=0.01, l1_cutoff=0.02)

print(len(dup_group))
print("dinfty_cutoff=0.01, d1_cutoff=0.02")
print("elements set of 10 largest groups:")
for i in pd.Series(group_rows).value_counts().index[:10]:
    print(f"{hsd._df.loc[i, 'elements_set']} shown {len(dup_group[i])} times. indices: {dup_group[i]}")

#%% 
# test group - to prevent data leakage (inbetween train-test split: dinfty_cutoff=0.05, d1_cutoff=0.12) 
dup_group, group_rows = make_duplicates_group(hsd._df.index,
                                              l1_dist_matrix, linfty_dist_matrix,
                                              elements_sets,
                                              linfty_cutoff=0.05, l1_cutoff=0.12,
                                              cross_elements_set=True)

print(len(dup_group))
print("dinfty_cutoff=0.05, d1_cutoff=0.12")
print("elements set of 10 largest group:")
for i in pd.Series(group_rows).value_counts().index[:10]:
    print(f"{hsd._df.loc[i, 'elements_set']} shown {len(dup_group[i])} times. indices: {dup_group[i]}")

# %% 
# # compare with old method - BaseDataset.pymatgen_duplicates()

linfty_cutoff=0.01
l1_cutoff=0.02
print(f"dinfty_cutoff={linfty_cutoff}, d1_cutoff={l1_cutoff}")
dup_group, group_rows = make_duplicates_group(hsd._df.index, 
                                              l1_dist_matrix, linfty_dist_matrix, 
                                              elements_sets,
                                              linfty_cutoff=0.01, l1_cutoff=0.02)
assert len(dup_group)==361

# pymatgen_duplicates():
#   - initializes `self.duplicated_comps_group=dict()` and `self.duplicated_comps=set()`.
#   - `self.duplicated_comps_group.keys()` are entry-index of group's first item.
hsd.pymatgen_duplicates() 
#%%
# checkout different groups;

def compare_dupl_groups(hsd, dup_group, group_rows):
    for old_group_index, old_group in hsd.duplicated_comps_group.items():
        if set(old_group.keys())!=set(dup_group[group_rows[old_group_index]]):
            print(f"old group[{old_group_index}] indices: {old_group.keys()}")
            print(f"new group[{group_rows[old_group_index]}] indices: {dup_group[group_rows[old_group_index]]}")

            new_l1_min=100.0
            new_linfty_min=100.0
            new_group_comps = [hsd[idx]['comps_pymatgen'] for idx in dup_group[group_rows[old_group_index]]]
            old_group_max_l1=0.0
            old_group_max_linfty=0.0
            old_l1_min=100.0
            old_linfty_min=100.0
            old_group_comps = [Composition(comp) for comp in hsd.duplicated_comps_group[old_group_index].values()]

            old_group_comps_only = set(old_group_comps) - set(new_group_comps) # composition only in new
            new_group_comps_only = set(new_group_comps) - set(old_group_comps) # composition only in old

            if len(new_group_comps_only)>0 or len(set(new_group_comps))>len(set(old_group_comps)):
                new_group_only_idxs = [new_group_comps.index(comp) for comp in new_group_comps_only]
                new_l1_matrix=distance_matrix(new_group_comps, metric='l1')
                new_linfty_matrix = distance_matrix(new_group_comps, metric='l_infty')
                print(f"l1 matrix on new group:\n{new_l1_matrix}")
                print(f"l_infty matrix on new group:\n{new_linfty_matrix}")
                new_l1_min=min(new_l1_matrix[new_group_only_idxs].min(), new_l1_min)
                new_linfty_min=min(new_linfty_matrix[new_group_only_idxs].min(), new_linfty_min)
                print("new groups are larger")

            if len(old_group_comps_only)>0:
                old_group_only_idxs = [old_group_comps.index(comp) for comp in old_group_comps_only]
                old_l1_matrix=distance_matrix(old_group_comps, metric='l1')
                old_linfty_matrix=distance_matrix(old_group_comps, metric='l_infty')
                print(f"l1 matrix on old group:\n{old_l1_matrix}")
                print(f"l_infty matrix on old group:\n{old_linfty_matrix}")
                old_group_max_l1=max(old_l1_matrix[old_group_only_idxs].max(), old_group_max_l1)
                old_group_max_linfty=max(old_linfty_matrix[old_group_only_idxs].max(), old_group_max_linfty)
                for i in range(len(old_l1_matrix)):
                    for j in range(len(old_l1_matrix[0])):
                        if old_l1_matrix[i, j]<=l1_cutoff and old_linfty_matrix[i, j]<=linfty_cutoff: #grouped case
                            print("asdfdsf")
                            old_l1_min=min(old_l1_matrix[i, j], old_l1_min)
                            old_linfty_min=min(old_linfty_matrix[i,j], old_linfty_min)
    return dict(
        old_l1_min = old_l1_min,
        old_linfty_min= old_linfty_min,
        new_l1_min=new_l1_min,
        new_linfty_min=new_linfty_min,
        old_group_max_l1=old_group_max_l1,
        old_group_max_linfty=old_group_max_linfty
    )
                    
                    
# %%
linfty_cutoff=0.08
l1_cutoff=0.02
print(f"dinfty_cutoff={linfty_cutoff}, d1_cutoff={l1_cutoff}")
dup_group, group_rows = make_duplicates_group(hsd._df.index, 
                                              l1_dist_matrix, linfty_dist_matrix, 
                                              elements_sets,
                                              linfty_cutoff=linfty_cutoff, l1_cutoff=l1_cutoff)

a = compare_dupl_groups(hsd, dup_group, group_rows)