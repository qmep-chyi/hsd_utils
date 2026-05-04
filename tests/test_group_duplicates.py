"""package's optional test module for group duplicates

        * representative results:
            * elements set `Hf-Zr-Ti-Ta-Nb` with 43 entries, largest duplicates group is group2: 11 entries with elemental set: Hf-Zr-Ti-Ta-Nb
            * last output: among 45 entries of XuTestHEA(), no duplicates in HE-SC dataset:[7, 20, 34, 40]

main features:
    * group close compositions

    $ uv run python -m unittest test_group_duplicates
    
"""

import unittest
import warnings

import pandas as pd

from hsdu.dataset import Dataset
from hsdu.comparison import XuTestHEA
from .utils_for_test import TestSnapshotWarning, get_package_dataset


class TestGroupDuplicates(unittest.TestCase):
    """Test core class methods"""

    def test_group_duplicates(self):
        """test duplicated entries(have too close compositions)
        """
        with get_package_dataset() as path:
            test_dataset_path = path
        hsd = Dataset(test_dataset_path, config="default.json")

        assert hsd._df.index.tolist()==list(range(len(hsd)))
        elements_sets = [hsd[i]['elements_set'] for i in hsd._df.index]
        
        print("count of elements_sets:")
        count_of_elements_sets = pd.Series(elements_sets).value_counts()
        print(count_of_elements_sets)
        if not count_of_elements_sets[0]==43: 
            warnings.warn(count_of_elements_sets, TestSnapshotWarning)
        if not count_of_elements_sets.index[0]=='Hf-Zr-Ti-Ta-Nb':
            warnings.warn(count_of_elements_sets.index[0], TestSnapshotWarning)

        # internal duplicates
        dup_group, idx2group = hsd.group_duplicates(cityblock=0.01, msre=0.02)

        print(f'number of internal duplicates groups (including group with only 1 entry):{len(dup_group)}')
        print('largest groups:')
        largest_groups = pd.Series(idx2group).value_counts()[:10]
        elemental_sets_of_largest_groups = [elements_sets[list(dup_group[i])[0]] for i in largest_groups.index]
        for (group_idx, count), elem_set in zip(largest_groups.items(), elemental_sets_of_largest_groups):
            print(f'group {group_idx}: {count} entries with elemental set:{elem_set}')
        if not largest_groups.index[0]==2 or not elemental_sets_of_largest_groups[0]=='Hf-Zr-Ti-Ta-Nb':
            warnings.warn((largest_groups.index[0], elemental_sets_of_largest_groups[0]), TestSnapshotWarning)

        # to other datatable
        xu_dataset = XuTestHEA()
        xu_dataset.encode_onehot_fracs(fixed_elements_set=hsd.elemental_set, rule_elements_set='overwrite', composition_col='formula')
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

#if __name__=="__main__":
    #unittest.main()
    #TestGroupDuplicates("test_group_duplicates").debug()