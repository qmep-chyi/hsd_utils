"""package's optional test module for group duplicates

main features:
    * group close compositions

    $ uv run python -m unittest test_group_duplicates
    
"""

import importlib.resources as resources
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Literal
from scipy.spatial.distance import cdist
from pymatgen.core import Composition

from hsdu.dataset import Dataset, D2TableDataset
from hsdu.utils.conversion_utils import almost_equals_pymatgen_atomic_fraction, element_list_iupac_ordered
from hsdu.utils.duplicate import make_duplicates_group, distance_matrix, compare_dupl_groups, compare_dupl_groups_old2new, dist4groups_matrix
from hsdu.comparison import XuTestHEA, StanevSuperCon


class TestGroupDuplicates(unittest.TestCase):
    """Test core class methods"""

    def test_group_duplicates(self):
        """test duplicated entries(have too close compositions)
        """
        test_dataset_path: Path=None
        with resources.as_file(resources.files("hsdu.data.tests") /"full_dataset.csv") as path:
            test_dataset_path = path if Path(path).is_file() else None

        if test_dataset_path is None:
            with resources.as_file(resources.files("hsdu.data.tests") /"test_dataset.csv") as path:
                test_dataset_path = path
        hsd = Dataset(test_dataset_path, config="default.json")

        assert hsd._df.index.tolist()==list(range(len(hsd)))
        elements_sets = [hsd[i]['elements_set'] for i in hsd._df.index]
        print(pd.Series(elements_sets).value_counts())

        # internal duplicates
        dup_group, idx2group = hsd.group_duplicates(cityblock=0.01, msre=0.02)

        print(f'number of internal duplicates groups (including group with only 1 entry):{len(dup_group)}')
        print('largest groups:')
        largest_groups = pd.Series(idx2group).value_counts()[:10]
        elemental_sets_of_largest_groups = [elements_sets[list(dup_group[i])[0]] for i in largest_groups.index]
        for (group_idx, count), elem_set in zip(largest_groups.items(), elemental_sets_of_largest_groups):
            print(f'group {group_idx}: {count} entries with elemental set:{elem_set}')

        # to other datatable
        xu_dataset = XuTestHEA()
        xu_dataset.encode_onehot_fracs(fixed_elements_set=hsd.elemental_set, rule_elements_set='overwrite', parse_pymatgen_comps_col='formula')
        dupl_group, idx2group=hsd.group_duplicates(other=xu_dataset, cityblock=0.01, msre=0.02, update_attrs=False)
        print('group_duplicates with XuTestHEA()')
        existing_dupl_groups=dict()
        group_indices=[]
        for k, v in dupl_group.items():
            if len(v)>0:
                existing_dupl_groups[k]=v
                group_indices.append(k)
        missing_index=[]
        for i in range(len(xu_dataset)):
            if i not in group_indices:
                missing_index.append(i)
        print(f'among {len(xu_dataset)} entries of XuTestHEA(), no duplicates in HE-SC dataset:{missing_index}')

if __name__=="__main__":
    #unittest.main()
    TestGroupDuplicates("test_group_duplicates").debug()