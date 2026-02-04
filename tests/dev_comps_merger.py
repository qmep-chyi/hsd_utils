#%%
import numpy as np
import pandas as pd
from typing import Literal

from scipy.spatial.distance import cdist
from pymatgen.core import Composition

from hsdu.dataset import Dataset
from hsdu.utils.conversion_utils import almost_equals_pymatgen_atomic_fraction, element_list_iupac_ordered
from hsdu.utils.duplicate import make_duplicates_group, distance_matrix
#%%
hsdset_path=r"C:\Users\hms_l\hsd_utils\temp_devs\hesc_dataset_16dec2025 - DataTable.csv"
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
linfty_dist_matrix = distance_matrix(onehot_fracs,, metric="l_infty", elemental_set=[row["elements_set"] for row in hsd])
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
# # compare with old method - BaseDataset.pymatgen_duplicates()

linfty_cutoff=0.01
l1_cutoff=0.02
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

def dist4groups_matrix(groups_of_vectors:list[list[list[float]]], metric:Literal["cityblock", "chebyshev"]):
    out = np.zeros((len(groups_of_vectors),len(groups_of_vectors)))
    for group_i, vectors0 in enumerate(groups_of_vectors):
        for group_j, vectors1 in enumerate(groups_of_vectors):
            if group_i==group_j:
                pass
            else:
                out[group_i, group_j] = cdist(vectors0, vectors1, metric=metric).min()
    return out

def compare_dupl_groups(dataset, dup_group, group_rows):
    disagreement=False
    for old_group_index, old_group in dataset.duplicated_comps_group.items():
        if set(old_group.keys())!=set(dup_group[group_rows[old_group_index]]):
            disagreement=True
            print(f"old group[{old_group_index}] indices: {old_group.keys()}")
            assert old_group_index in old_group.keys()
            new_groups_overlapped = list(set([i for i in group_rows[old_group.keys()]]))
            print(f"new groups overlapped: [{new_groups_overlapped}] indices: {[dup_group[i] for i in new_groups_overlapped]}")
            
            relative_groups_name = [f"old_group{old_group_index}"]+new_groups_overlapped

            relative_groups=[[dataset[idx]['comps_pymatgen'] for idx in old_group.keys()]]
            for new_group_idx in new_groups_overlapped:
                for new_group_idx in dup_group[new_group_idx]:
                    relative_groups=relative_groups+[dataset[i]['comps_pymatgen'] for i in new_group_idx]
            
            l1_dist4groups_matrix=dist4groups_matrix(relative_groups, metric="cityblock")
            l1_dist4groups_matrix=pd.DataFrame(l1_dist4groups_matrix, 
                                               columns=relative_groups_name, index=relative_groups_name)
            linfty_dist4groups_matrix=dist4groups_matrix(relative_groups, metric="chebyshev")
            linfty_dist4groups_matrix=pd.DataFrame(linfty_dist4groups_matrix,
                                                   columns=relative_groups_name, index=relative_groups_name)
            
            if False:
                for new_group_idx in new_groups_overlapped:
                    #min_l1_inbetween=100.0
                    new_l1_min=100.0
                    new_linfty_min=100.0
                    new_group_comps = [dataset[idx]['comps_pymatgen'] for idx in dup_group[new_group_idx]]
                    old_group_max_l1=0.0
                    old_group_max_linfty=0.0
                    old_l1_min=100.0
                    old_linfty_min=100.0
                    old_group_comps = [Composition(comp) for comp in dataset.duplicated_comps_group[old_group_index].values()]

                    old_group_only_comps = set(old_group_comps) - set(new_group_comps) # composition only in new
                    new_group_only_comps = set(new_group_comps) - set(old_group_comps) # composition only in old

                    if len(new_group_only_comps)>0 or len(set(new_group_comps))>len(set(old_group_comps)):
                        new_group_only_idxs = [new_group_comps.index(comp) for comp in new_group_only_comps]
                        new_l1_matrix=distance_matrix(new_group_comps, metric='l1')
                        new_linfty_matrix = distance_matrix(new_group_comps, metric='l_infty')
                        print(f"l1 matrix on new group:\n{new_l1_matrix}")
                        print(f"l_infty matrix on new group:\n{new_linfty_matrix}")
                        new_l1_min=min(new_l1_matrix[new_group_only_idxs].min(), new_l1_min)
                        new_linfty_min=min(new_linfty_matrix[new_group_only_idxs].min(), new_linfty_min)
                        print("new groups are larger")

                    if len(old_group_only_comps)>0:
                        old_group_only_idxs = [old_group_comps.index(comp) for comp in old_group_only_comps]
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
    if disagreement:
        return dict(
            linfty_dist4groups_matrix = linfty_dist4groups_matrix,
            l1_dist4groups_matrix= l1_dist4groups_matrix)
                    
                    
# %%
linfty_cutoff=0.08
l1_cutoff=0.02
dup_group, group_rows = make_duplicates_group(hsd._df.index, 
                                              l1_dist_matrix, linfty_dist_matrix, 
                                              elements_sets,
                                              linfty_cutoff=linfty_cutoff, l1_cutoff=l1_cutoff)

a = compare_dupl_groups(hsd, dup_group, group_rows)