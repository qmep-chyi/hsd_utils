#%%


import pandas as pd

from scipy.spatial.distance import cdist

from hsdu.dataset import Dataset
from hsdu.utils.conversion_utils import almost_equals_pymatgen_atomic_fraction, element_list_iupac_ordered
#%%
hsdset_path="/home/chyi/hsd_utils/tests/temp_devs/hesc_dataset_16dec2025 - DataTable.csv"
hsd = Dataset(hsdset_path)

#%%
# test - encode / decode to comps_onehot_frac
# ####TODO:#### update these list to the dataset class's attribute

onehot_fracs = []
for idx, row in hsd.df.iterrows():
    a= hsd.pymatgen_comps(idx)
    onehot_frac = hsd.onehot_frac.encode(a)
    d=hsd.onehot_frac.decode(onehot_frac)
    onehot_fracs.append(onehot_frac)
    assert almost_equals_pymatgen_atomic_fraction(a, d, rtol=1e-9)
#%%
# Chebyshev
# chebyshev_dist = $L_{\infty}$=$\max_i |a_i - b_i|$ test
print(f'frac 0:{hsd.df.loc[[0,1],"elements"]}{hsd.df.loc[[0,1],"elements_fraction"]}')
print(f'pymatgen_comps(0):{hsd.pymatgen_comps(0)}')
print(f'pymatgen_comps(1):{hsd.pymatgen_comps(1)}')
onehot_frac0 = hsd.onehot_frac.encode(hsd.pymatgen_comps(0))
onehot_frac1 = hsd.onehot_frac.encode(hsd.pymatgen_comps(1))

print(cdist([onehot_frac0],[onehot_frac1],"chebyshev"))
# L1 dist
print(cdist([onehot_frac0],[onehot_frac1],"cityblock"))
#%%
# make elements_set list
elements_sets=[]
for i in hsd.df.index:
    elements_sets.append("-".join(element_list_iupac_ordered(hsd.pymatgen_comps(i).elements)))
print(pd.Series(elements_sets).value_counts())
#%%
# make distance matrix

# chebyshev distance
dinfty_dist_matrix = cdist(onehot_fracs, onehot_fracs, metric="chebyshev")
# L1 distance
d1_dist_matrix = cdist(onehot_fracs, onehot_fracs, metric="cityblock")
#%% to merge entries with close compositions
    # merge_overlap
def merge_overlap(index, duplicates_group, group_rows):
    overlaps=False
    for i in index:
        first_group=-1
        other_groups=[]
        for k, v in duplicates_group.items():
            if i in v:
                if first_group==-1:
                    first_group=k
                else:
                    other_groups.append(k)
        if len(other_groups)>0:
            overlaps=True
            for j in other_groups:
                for k in duplicates_group[j]:
                    duplicates_group[first_group].append(k)
                    group_rows[k]=first_group
                duplicates_group.pop(j)
    
    duplicates_group = {k: list(set(v)) for k, v in duplicates_group.items()}

    if overlaps:
        return merge_overlap(index, duplicates_group, group_rows)
    else:
        assert sum(len(set(v)) for v in duplicates_group.values()) == len(index)
        for i in index:
            assert sum([1 if i in v else 0 for v in duplicates_group.values()])==1
        return duplicates_group, group_rows
        
def make_duplicates_group(index, d1_dist_matrix, dinfty_dist_matrix, elements_sets, dinfty_cutoff=0.01, d1_cutoff=0.02, cross_elements_set=False):
    assert index.tolist()==list(range(len(elements_sets)))
    last_dup_group_idx = -1 # so first is 0
    current_group_idx = 0
    duplicates_group = dict()
    group_rows = dict()

    for i in index:
        # init group
        if group_rows.get(i) is None: 
            last_dup_group_idx+=1
            duplicates_group.setdefault(last_dup_group_idx, []).append(i)
            group_rows.setdefault(i, last_dup_group_idx)
            current_group_idx = last_dup_group_idx
        else:
            current_group_idx = group_rows[i]
        # process duplicates
        for j in index:
            if d1_dist_matrix[i, j]<d1_cutoff and dinfty_dist_matrix[i, j]<dinfty_cutoff:
                if cross_elements_set:
                    if j<=i: # just double check
                        assert i in duplicates_group[group_rows[j]]
                    elif j>i:
                        duplicates_group[current_group_idx].append(j)
                        group_rows[j]=current_group_idx
                    else:
                        raise ValueError
                else:
                    if elements_sets[i]==elements_sets[j]:
                        if j<=i: # just double check
                            assert i in duplicates_group[group_rows[j]]
                        elif j>i:
                            duplicates_group[current_group_idx].append(j)
                            group_rows[j]=current_group_idx
                        else:
                            raise ValueError
        duplicates_group = {k: list(set(v)) for k, v in duplicates_group.items()}
            
    return merge_overlap(index, duplicates_group, group_rows)


assert hsd.df.index.tolist()==list(range(len(hsd)))
dup_group, group_rows = make_duplicates_group(hsd.df.index, d1_dist_matrix, dinfty_dist_matrix, elements_sets, dinfty_cutoff=0.01, d1_cutoff=0.02)

print(len(dup_group))
print(pd.Series(group_rows).value_counts())
print([elements_sets[dup_group[i][0]] for i in pd.Series(group_rows).value_counts().index[:10]])


#%% prevent data leakage (train-test split)

dup_group, group_rows = make_duplicates_group(hsd.df.index, d1_dist_matrix, dinfty_dist_matrix, elements_sets, dinfty_cutoff=0.05, d1_cutoff=0.12, cross_elements_set=True)

print(len(dup_group))
print(pd.Series(group_rows).value_counts())
