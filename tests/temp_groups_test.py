#%%
import numpy as np
import pandas as pd
from typing import Literal

from scipy.spatial.distance import cdist
from pymatgen.core import Composition

from hsdu.dataset import Dataset, D2TableDataset
from hsdu.utils.conversion_utils import almost_equals_pymatgen_atomic_fraction, element_list_iupac_ordered
from hsdu.utils.duplicate import make_duplicates_group, distance_matrix, compare_dupl_groups, compare_dupl_groups_old2new


# %% # TODO: add test relate to this cell or just delete these?
# ##### Compare 251127 train-test set ######
#test_path = "afdgafgg"
#train_path = "hghg"
#testset = D2TableDataset(test_path, exception_col=None, drop_cols=['comps_pymatgen'])
#trainset = D2TableDataset(train_path, exception_col=None, drop_cols=['comps_pymatgen'])

#test_comps = [row['comps_pymatgen'] for row in testset]
#train_comps = [row['comps_pymatgen'] for row in trainset]

#train_test_comparison=compare_dupl_groups([train_comps, test_comps], ['train', 'test'], ignore_cross_elemental_set=False)

# %% # TODO: refactor
# ##### continue implement duplicated group ######
# first copied from hsdu.utils.duplicate test codes. delete them (under `if __name__=="__main__"`) after this.

hsd_path = r''
hsd = Dataset(hsd_path, exception_col='Exceptions')

assert hsd._df.index.tolist()==list(range(len(hsd)))
elements_sets = [hsd[i]['elements_set'] for i in hsd._df.index]
print(pd.Series(elements_sets).value_counts())

# make distance matrix
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

dup_group, group_rows = make_duplicates_group(hsd._df.index, d1_dist_matrix, dinfty_dist_matrix, elements_sets, linfty_cutoff=0.01, l1_cutoff=0.02)

print(len(dup_group))
print(pd.Series(group_rows).value_counts())
print([elements_sets[list(dup_group[i])[0]] for i in pd.Series(group_rows).value_counts().index[:10]])

#%% test to prevent data leakage (train-test split)

dup_group, group_rows = make_duplicates_group(hsd._df.index, d1_dist_matrix, dinfty_dist_matrix, elements_sets, linfty_cutoff=0.05, l1_cutoff=0.12, cross_elements_set=True)