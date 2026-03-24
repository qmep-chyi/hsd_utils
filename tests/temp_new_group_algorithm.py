# ###### validated new group algorithm #####
# ***exported from jupyter notebook, didn't tried to run this***
# TODO: 
#   * replace old duplicate group method with new one
#       * old: hsdu.data.Dataset().pymatgen_duplicates(rtol=0.02)
#   * Then, refactor these; to test and document.
#       - temp_new_group_algorithm
#      - dev_comps_merger.py
#       - temp_groups_test.py

# %% [markdown]
# # Validation of duplicates groups
# ## Summary
# ### Close compositions
# * in train-test split (16 Dec 2025)
# 
# |composition|T_c(K)|set|
# |---|---|---|
# | $\textrm{Ta}_{0.335}\textrm{Nb}_{0.335}\textrm{Hf}_{0.11}\textrm{Zr}_{0.11}\textrm{Ti}_{0.11}$ | 8.03 | test |
# | $\textrm{Ta}_{0.35}\textrm{Nb}_{0.35}\textrm{Hf}_{0.1}\textrm{Zr}_{0.1}\textrm{Ti}_{0.1}$ | 7.75 | train |
# | $\textrm{Mo}_{0.105}\textrm{Re}_{0.105}\textrm{Ru}_{0.527}\textrm{Rh}_{0.10}\textrm{Ti}_{0.158}$ | 2.1 | test |
# | $\textrm{Mo}_{0.1}\textrm{Re}_{0.1}\textrm{Ru}_{0.55}\textrm{Rh}_{0.10}\textrm{Ti}_{0.15}$ | 2.2 | train |
# 
# * Examined `duplicates_merge` process (about data leakage)
#     * validated with `xgboost regressor`
#     * grouped split, repeated CV
#     * To determine interpolation / extrapolation performance

# %% [markdown]
# ### Group algorithms
# * Previous group algorithm (since 22 Sep 2025)
#     * metric: max relative error (MRE):
#         * $L(x,y)=max_i(|x_i-y_i|/y_i)$. So, $L(x,y)\neq L(y,x)$
#         * where $x_i$, $y_i$ are fraction of i-th element
#     * groupped if $L<0.02$
#         * If two composition have same elements
#         * Once groupped, fix the groups.
#         * e.g. $L(A,C),L(B,C)<0.02$ but $L(A, B)>0.02$ -> group (A, C) only
#     * composition and $T_c$ of merged entry:
#         * composition: which had $max(T_c)$
#         * $T_c$:
#             * re-calculated `max`, `min`, `avg` when merging
#             * used `avg` for prediction target usually
# * New distance metrics tested
#     1. L1 distance(Mean Absolute Error) $L_1(x,y)=\sum_i{|x_i-y_i|}$
#     2. $L_\infty$ (Chebyshev) distance: $L_\infty(x,y)=max_i|x_i-y_i|$
#     3. MSRE (max symmetric relative error): $max(MRE(x,y), MRE(y,x))$
# * New group algoritm:
#     * Recursively merge groups if $min_{i, j}L(A_i, B_j) < L_{cutoff}$ where A, B are groups.
#     * so, it is possible to be $L_{i,j}>L_{cutoff}$ among a group
#     * (For validation only, not used to merge duplicates), optionally groups even if $A.elements \neq B.elements$
# 
# ### Reproduce old groups with new metric;
# * same if MSRE $<0.011$—MSRE $<0.02$ (2% difference)
# * Almost similar if L1 $<0.01$ except for 8 entries
#     * old 6 groups merged to 3 groups in this case
# 
# ### supplement: nominal - actual compositions
# |idx (16dec)|elements|fractions|
# |---|---|---|
# |88|['Ag', 'Sn', 'Pb', 'Bi', 'Te']|{"nominal": [0.25, 0.25, 0.25, 0.25, 1], "actual": [0.27, 0.26, 0.22, 0.25, 1]}|
# |89|['Ag', 'Sn', 'Pb', 'Bi', 'In', 'Te']|{"nominal": [0.225, 0.225, 0.225, 0.225, 0.10, 1], "actual": [0.24, 0.23, 0.21, 0.22, 0.1, 1]}|
# |90|['Ag', 'Sn', 'Pb', 'Bi', 'In', 'Te']|{"nominal": [0.20, 0.20, 0.20, 0.20, 0.20, 1], "actual": [0.20, 0.22, 0.19, 0.19, 0.20, 1]}|
# |91|['Ag', 'Sn', 'Pb', 'Bi', 'In', 'Te']|{"nominal": [0.175, 0.175, 0.175, 0.175, 0.30, 1], "actual": [0.17, 0.19, 0.15, 0.18, 0.31, 1]}|
# |92|['Ag', 'Sn', 'Pb', 'Bi', 'In', 'Te']|{"nominal": [0.15, 0.15, 0.15, 0.15, 0.40, 1], "actual": [0.15, 0.16, 0.15, 0.14, 0.40, 1]}|
# |108|['Sc', 'Hf', 'Nb', 'Ti']|{"nominal": [0.25, 0.25, 0.25, 0.25], "bcc": [0.20, 0.25, 0.30, 0.25], "hcp": [0.33, 0.22, 0.22, 0.23]}|
# |109|['Sc', 'Hf', 'Nb', 'Zr']|{"nominal": [0.25, 0.25, 0.25, 0.25], "bcc": [0.21, 0.26, 0.28, 0.25], "hcp": [0.28, 0.21, 0.25, 0.26]}|
# |110|['Sc', 'Hf', 'Ta', 'Ti']|{"nominal": [0.25, 0.25, 0.25, 0.25], "bcc": [0.12, 0.24, 0.38, 0.26], "hcp": [0.56, 0.20, 0.08, 0.16]}|
# |111|['Sc', 'Nb', 'Ti', 'Zr']|{"nominal": [0.25, 0.25, 0.25, 0.25], "bcc": [0.18, 0.32, 0.24, 0.26], "hcp": [0.33, 0.20, 0.21, 0.26]}|
# |112|['Sc', 'Ta', 'Ti', 'Zr']|{"nominal": [0.25, 0.25, 0.25, 0.25], "bcc": [0.04, 0.64, 0.22, 0.10], "hcp": [0.32, 0.10, 0.26, 0.32]}|
# |113|['Sc', 'Hf', 'Nb', 'Ta', 'Ti']|{"nominal": [0.20, 0.20, 0.20, 0.20, 0.20], "bcc": [0.08, 0.22, 0.25, 0.25, 0.20], "hcp": [0.67, 0.14, 0.05, 0.04, 0.10]}|
# |114|['Sc', 'Hf', 'Nb', 'Ta', 'Zr']|{"nominal": [0.20, 0.20, 0.20, 0.20, 0.20], "bcc-I": [0.09, 0.21, 0.26, 0.26, 0.18], "bcc-II": [0.15, 0.22, 0.22, 0.20, 0.21], "hcp": [0.36, 0.19, 0.12, 0.08, 0.25]}|
# |115|['Sc', 'Nb', 'Ta', 'Ti', 'Zr']|{"nominal": [0.20, 0.20, 0.20, 0.20, 0.20], "bcc": [0.04, 0.29, 0.41, 0.15, 0.11], "hcp": [0.33, 0.12, 0.05, 0.19, 0.31]}|
# |116|['Sc', 'Hf', 'Nb', 'Ta', 'Ti', 'Zr']|{"nominal": ['1/6', '1/6', '1/6', '1/6', '1/6', '1/6'], "bcc": [0.12, 0.17, 0.19, 0.19, 0.17, 0.16], "hcp": [0.29, 0.15, 0.13, 0.12, 0.13, 0.18]}|
# |173|['Nb', 'Re', 'Zr', 'Hf', 'Ti']|{"nominal":[0.2, 0.2, 0.2, 0.2, 0.2], "actual":[0.21, 0.16, 0.20, 0.23, 0.20]}|
# |324|['Nb', 'Ta', 'Ti', 'Zr', 'Hf']|{"nominal":[0.2, 0.2, 0.2, 0.2, 0.2], "dendrite":[0.221, 0.263, 0.166, 0.155, 0.195]}|
# |326|['Nb', 'Ta', 'Ti', 'Zr', 'Hf', 'V']|{"nominal":["1/6", "1/6", "1/6", "1/6", "1/6", "1/6"], "dendrite":[0.215, 0.181, 0.159, 0.144, 0.166, 0.135]}|
# |400|['Mo', 'Re', 'Ru', 'Pd', 'Pt']|{"nominal": [0.1113, 0.1113, 0.1113, 0.333, 0.333], "actual": [0.109, 0.096, 0.135, 0.317, 0.343]}|
# 
# ### Miscs
# * old versions
#     * [30jan_xgboost](https://colab.research.google.com/drive/1zBp6tnfwSrZ5XLNtHQxkYMz9TdDmZ6oQ#scrollTo=KJRqh3d-B398)
#     * [06jan_xgboost](https://colab.research.google.com/drive/1IDAVA2dcIZ_SMuh-f2w8PhKG2fWssaqg#scrollTo=gPaaa_T3Yv3v)

# %% [markdown]
# ## Environment, import, load dataset
# Neccessary run

# %%
import numpy as np
import pandas as pd
from typing import Literal

from scipy.spatial.distance import cdist
from pymatgen.core import Composition

from hsdu.dataset import Dataset, D2TableDataset
from hsdu.utils.conversion_utils import almost_equals_pymatgen_atomic_fraction, element_list_iupac_ordered
from hsdu.utils.duplicate import make_duplicates_group, distance_matrix, compare_dupl_groups, compare_dupl_groups_old2new, dist4groups_matrix

dataset_path = r"/home/chyi/hsd_utils/src/hsdu/data/tests/full_dataset.csv"
hsd = Dataset(dataset_path)

# %%
import scipy
print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"scipy: {scipy.__version__}")
print("pymatgen")
!uv pip show pymatgen

# %%
# calculate distances as matrix
onehot_fracs = hsd.onehot_fracs()
dist_matrices=dict(
    chebyshev=distance_matrix(onehot_fracs, onehot_fracs,
                           metric="l_infty",
                           elemental_set=hsd.elemental_set),
    cityblock = distance_matrix(onehot_fracs, onehot_fracs,
                         metric="l1",
                         elemental_set=hsd.elemental_set),
    MSRE = distance_matrix(onehot_fracs, onehot_fracs,
                           metric='max_sym_relative_error',
                           elemental_set=hsd.elemental_set)
)

elements_sets_rows = [hsd[i]["elements_set"] for i in hsd.index]


# %% [markdown]
# ## New group algorithm

# %% [markdown]
# ### Elements sets

# %%
assert hsd.index==list(range(len(hsd)))

print(pd.Series(elements_sets_rows).value_counts().head(20))

# %% [markdown]
# ### Load old group

# %%
# prepare old groups;
# pymatgen_duplicates():
#   - initializes `self.duplicated_comps_group=dict()` and `self.duplicated_comps=set()`.
#   - `self.duplicated_comps_group.keys()` are entry-index of group's first item.
hsd.pymatgen_duplicates(rtol=0.02)
hsd.duplicated_comps_group

# %%
# old group size;
print(len(hsd.duplicated_comps_group)) # number of duplicate group
print(len(hsd.duplicated_comps)) #number of duplicates entries
hsd.duplicated_comps_group

# %% [markdown]
# ## New groups
# ### distances
# 

# %%
# calculate distances as matrix
onehot_fracs = hsd.onehot_fracs()

# %% [markdown]
# example case;
# |composition|$T_c (K)|set|
# |----|----|----|
# |${Ta_{0.335}Nb_{0.335}Hf_{0.11}Zr_{0.11}Ti_{0.11}}$|8.03|test|
# |${Ta_{0.35}Nb_{0.35}Hf_{0.1}Zr_{0.1}Ti_{0.1}}$|7.75|train|
# |${Mo_{0.105}Re_{0.105}Ru_{0.527}Rh_{0.10}Ti_{0.158}}$|2.1|test|
# |${Mo_{0.1}Re_{0.1}Ru_{0.55}Rh_{0.10}Ti_{0.15}}$|2.2|train|

# %%
# exapmle distances:
a = [
    [0.335, 0.335, 0.11, 0.11, 0.11, 0.0, 0.0, 0.0, 0.0],
    [0.35, 0.35, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.158, 0.105, 0.105, 0.527, 0.10],
    [0.0, 0.0, 0.0, 0.0, 0.15, 0.1, 0.1, 0.55, 0.10]
    ]
print("Example distance matrix")
print(f"L_infty: \n{distance_matrix(a, a, metric='chebyshev')}")
print(f"L_1: \n{distance_matrix(a, a, metric='cityblock')}")

# %% [markdown]
# ### make groups
# *  maximum symmetric relative error (MSRE)$<0.02$
#     * MSRE defined as $max(MRE(x,y), MRE(y,x))$
#     * where maximum relative error (MRE)$=max_i |x_i-y_i|/y_i$ if $y_i\neq 0$ else 0
# * $\textrm{L}_1<0.01$

# %%

dist_cutoffs=dict(cityblock=0.01, # L1 distance
                  MSRE=0.02)

dup_group, entry2group_idx = make_duplicates_group(hsd._df.index,dist_matrices,
                                                   elements_sets_rows, dist_cutoffs)

# %%
dup_group_wo_len1=dict()
for k, v in dup_group.items():
    if len(v)>1:
        dup_group_wo_len1[k]=v
dup_group_wo_len1

# %% [markdown]
# #### distance inbetween new groups
# * Closest (include cross-elemental set cases)

# %%
# distance inbetween new groups
group_comps = []
for dup_idx_set in dup_group.values():
    group_comps.append([hsd[dup_idx]['comps_pymatgen'] for dup_idx in dup_idx_set])

new_groups_distance_matrix=compare_dupl_groups(group_comps, dup_group.keys())
for dist_metric in dist_cutoffs.keys():
    dist_df = new_groups_distance_matrix[dist_metric]
    min_dist = np.nanmin(dist_df)
    mask=(dist_df == min_dist)
    print(f"min_{dist_metric}: {min_dist}")

    other_metrices = set(dist_cutoffs.keys())
    other_metrices.remove(dist_metric)
    group0_idx, group1_idx = dist_df.index[mask.any()]

    for other_metric in other_metrices:
        other_metric_val = compare_dupl_groups([[hsd[i]['comps_pymatgen'] for i in dup_group[group0_idx]],[hsd[i]['comps_pymatgen'] for i in dup_group[group1_idx]]],
                                            group_names=[group0_idx, group1_idx], print_output=False)
        other_metric_val_np = other_metric_val[other_metric].to_numpy().flatten()
        other_metric_val = set(other_metric_val_np[~np.isnan(other_metric_val_np)])
        assert len(other_metric_val)==1
        print(f"where {other_metric}: {other_metric_val}")
    print(f"group 0: {[hsd[idx]['comps_pymatgen'] for idx in dup_group[group0_idx]]}")
    print(f"group 1: {[hsd[idx]['comps_pymatgen'] for idx in dup_group[group1_idx]]}")


# %% [markdown]
# * Closest (inside the same elemental sets)

# %%
new_groups_distance_matrix=compare_dupl_groups(group_comps, dup_group.keys(), ignore_cross_elemental_set=True)
for dist_metric in dist_cutoffs.keys():
    dist_df = new_groups_distance_matrix[dist_metric]
    min_dist = np.nanmin(dist_df)
    mask=(dist_df == min_dist)
    print(f"min_{dist_metric}: {min_dist}")

    other_metrices = set(dist_cutoffs.keys())
    other_metrices.remove(dist_metric)
    group0_idx, group1_idx = dist_df.index[mask.any()]

    for other_metric in other_metrices:
        other_metric_val = compare_dupl_groups([[hsd[i]['comps_pymatgen'] for i in dup_group[group0_idx]],[hsd[i]['comps_pymatgen'] for i in dup_group[group1_idx]]],
                                            group_names=[group0_idx, group1_idx], print_output=False)
        other_metric_val_np = other_metric_val[other_metric].to_numpy().flatten()
        other_metric_val = set(other_metric_val_np[~np.isnan(other_metric_val_np)])
        assert len(other_metric_val)==1
        print(f"where {other_metric}: {other_metric_val}")
    print(f"group 0: {[hsd[idx]['comps_pymatgen'] for idx in dup_group[group0_idx]]}")
    print(f"group 1: {[hsd[idx]['comps_pymatgen'] for idx in dup_group[group1_idx]]}")

# %% [markdown]
# ## New -vs- old groups

# %% [markdown]
# ### group distance: inbetween old groups

# %%
hsd.pymatgen_duplicates(rtol=0.02)
old_group_comps = [[hsd[i]['comps_pymatgen'] for i in v.keys()] for k, v in hsd.duplicated_comps_group.items()]

new_groups_distance_matrix=compare_dupl_groups(old_group_comps, hsd.duplicated_comps_group.keys(), ignore_cross_elemental_set=True)

# %%
for dist_metric in dist_cutoffs.keys():
    dist_df = new_groups_distance_matrix[dist_metric]
    min_dist = np.nanmin(dist_df)
    mask=(dist_df == min_dist)
    print(f"min_{dist_metric}: {min_dist}")

    other_metrices = set(dist_cutoffs.keys())
    other_metrices.remove(dist_metric)
    group0_idx, group1_idx = dist_df.index[mask.any()]

    for other_metric in other_metrices:
        other_metric_val = compare_dupl_groups([[hsd[i]['comps_pymatgen'] for i in dup_group[group0_idx]],[hsd[i]['comps_pymatgen'] for i in dup_group[group1_idx]]],
                                            group_names=[group0_idx, group1_idx], print_output=False)
        other_metric_val_np = other_metric_val[other_metric].to_numpy().flatten()
        other_metric_val = set(other_metric_val_np[~np.isnan(other_metric_val_np)])
        assert len(other_metric_val)==1
        print(f"where {other_metric}: {other_metric_val}")
    print(f"group 0: {[hsd[idx]['comps_pymatgen'] for idx in dup_group[group0_idx]]}")
    print(f"group 1: {[hsd[idx]['comps_pymatgen'] for idx in dup_group[group1_idx]]}")

# %% [markdown]
# ### $l_\infty<0.01$ and $l_1<0.02$

# %%
dist_cutoffs=dict(cityblock=0.02, # L1 distance
                  chebyshev=0.01)

dup_group, entry2group_idx = make_duplicates_group(hsd._df.index,dist_matrices,
                                                   elements_sets_rows, dist_cutoffs)
print(len(dup_group))

# %%
# closest entries inbetween
old2new = compare_dupl_groups_old2new(hsd,dup_group, entry2group_idx)

# %% [markdown]
# ### New criteria matches with old group:
# * minimum distance should be splitted
#     * old_group 20 <-> old_group21
#         * $L_1 = 0.015254$
#         * $L_\infty = 0.003814$
#         * ***MSRE*** $=0.020483$
#     * entry 205, 209
#         * $L_1 = 0.006385$
#         * $L_\infty = 0.003192$
#         * ***MSRE*** $=0.028541$
#     * entry 345, 339
#         * $L_1 = 0.016667$
#         * $L_\infty = 0.008333$
#         * MSRE $=0.1$
# * So, $L_1 < 0.01 $ and MSRE $<0.02$ will work.

# %%
# groups should be splitted
# old_group 20 <-> old_group21
old_group20_comps = [hsd[i]['comps_pymatgen'] for i in [20, 264]]
old_group21_comps = [hsd[i]['comps_pymatgen'] for i in [21, 265]]
_ = compare_dupl_groups([old_group20_comps, old_group21_comps],(20, 21))

# 205, 209 nominal composition when (TiNbMoTaW)1-xNx , x=0.65 vs ~0.63(or 0.629)
comp_a = [hsd[i]['comps_pymatgen'] for i in [205]]
comp_b = [hsd[i]['comps_pymatgen'] for i in [209]]
_ = compare_dupl_groups([comp_a, comp_b],(205, 209))

# 345, 339
comp_a = [hsd[i]['comps_pymatgen'] for i in [345]]
comp_b = [hsd[i]['comps_pymatgen'] for i in [339]]
_ = compare_dupl_groups([comp_a, comp_b],(345, 339))

# %%
dist_cutoffs=dict(cityblock=0.01, # L1 distance
                  MSRE=0.02)
hsd.pymatgen_duplicates(rtol=0.02)
dup_group, idx2group_idx = make_duplicates_group(hsd._df.index,dist_matrices,
                                                   elements_sets_rows, dist_cutoffs,
                                                   cross_elements_set=False)
print(len(dup_group))
# closest entries inbetween

# %%
old2new = compare_dupl_groups_old2new(hsd,dup_group, idx2group_idx)

if compare_dupl_groups_old2new(hsd,dup_group, idx2group_idx) is None:
    print(f"if cutoff(threshold): {dist_cutoffs}, Exactly same with the old groups!")

# %%
hsd.pymatgen_duplicates(rtol=0.02)

dist_cutoffs=dict(#cityblock=0.01, # L1 distance
                  MSRE=0.02)
dup_group, idx2group_idx = make_duplicates_group(hsd._df.index,dist_matrices,
                                                   elements_sets_rows, dist_cutoffs,
                                                   cross_elements_set=False)

# %%
old2new = compare_dupl_groups_old2new(hsd,dup_group, idx2group_idx)

if compare_dupl_groups_old2new(hsd,dup_group, idx2group_idx) is None:
    print(f"if cutoff(threshold): {dist_cutoffs}, Exactly same with the old groups!")