#%%
import numpy as np
import pandas as pd
from typing import Literal

from scipy.spatial.distance import cdist
from pymatgen.core import Composition

from hsdu.dataset import Dataset, D2TableDataset
from hsdu.utils.conversion_utils import almost_equals_pymatgen_atomic_fraction, element_list_iupac_ordered
from hsdu.utils.duplicate import make_duplicates_group, distance_matrix, compare_dupl_groups, compare_dupl_groups_old2new

# %%
# ##### Compare 251127 train-test set ######
test_path = "afdgafgg"
train_path = "hghg"
testset = D2TableDataset(test_path, exception_col=None, drop_cols=['comps_pymatgen'])
trainset = D2TableDataset(train_path, exception_col=None, drop_cols=['comps_pymatgen'])

test_comps = [row['comps_pymatgen'] for row in testset]
train_comps = [row['comps_pymatgen'] for row in trainset]

train_test_comparison=compare_dupl_groups([train_comps, test_comps], ['train', 'test'], ignore_cross_elemental_set=False)