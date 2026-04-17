"""package's test module

main features:
    * load test version of in-house dataset
    * load additional dataset from literatures.
        * StanevSuperCon
        * XuDataset (see comparison.XuDataset for details)

    $ uv run python -m unittest test
    
"""

import unittest
import warnings

from hsdu.dataset import Dataset, D2TableDataset
from hsdu.comparison import XuTestHEA, StanevSuperCon
from .utils_for_test import TestSnapshotWarning, get_package_dataset


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        """in-house dataset test
        
        full_dataset.csv sholud exists in hsdu.data.tests
        """

        # load as a D2TableDataset

        dataset = D2TableDataset(get_package_dataset(), exception_col=None)

        # load dataset
        dataset = Dataset(get_package_dataset(), config="default.json")
        if dataset._df.shape!=(432, 105):
            warnings.warn(f"ss_dataset._df.shape={dataset._df.shape}",TestSnapshotWarning)
        else:
            print(f'dataset._df.shape: {dataset._df.shape}')
    def test_comparison(self):
        # load stanev's supercon dataset 
        ss_dataset = StanevSuperCon()
        ss_dataset.idx2aux['comps_pymatgen']
        
        if ss_dataset._df.shape!=(6194, 86):
            warnings.warn(f"ss_dataset._df.shape={ss_dataset._df.shape}",TestSnapshotWarning)
        else:
            print(f'ss_dataset._df.shape: {ss_dataset._df.shape}')

        #### Xu et al. 2025 test 45 hea set ####
        # gruop duplicates with HE-SC dataset
        xu_test_hea = XuTestHEA()
        if xu_test_hea._df.shape!=(45, 31):
            warnings.warn(f"xu_test45._df.shape=={xu_test_hea._df.shape}",TestSnapshotWarning)
        else:
            print(f'xu_test45._df.shape: {xu_test_hea._df.shape}')

        # laod HE-SC dataset for comparison
        with get_package_dataset() as path:
            test_dataset_path = path
        hsd = Dataset(test_dataset_path, config="default.json")

        # re-encode onhot fracs because elemental_set is different
        xu_test_hea.encode_onehot_fracs(fixed_elements_set=hsd.elemental_set, rule_elements_set='overwrite', parse_pymatgen_comps_col='formula')
        dupl_group_to_xu49, idx2group_to_xu49=hsd.group_duplicates(other=xu_test_hea, cityblock=0.01, msre=0.02, update_attrs=False)
        # cityblock=0.01, msre=0.02: group-criteria arbitrary chosen. 
        dupl_group_internal, idx2group_internal=hsd.group_duplicates(cityblock=0.01, msre=0.02)
        print('group_duplicates with XuTestHEA()')
        existing_dupl_groups=dict()
        group_indices=[]
        for k, v in dupl_group_to_xu49.items():
            if len(v)>0:
                existing_dupl_groups[k]=v
                group_indices.append(k)
        indices_not_found=[]
        for i in range(len(xu_test_hea)):
            if i not in group_indices:
                indices_not_found.append(i)
        print(f'among {len(xu_test_hea)} entries of XuTestHEA(), no duplicates in HE-SC dataset:{indices_not_found}')

        similar_entries_for_df=[]
        for idx, row in xu_test_hea._df.iterrows():
                hesc_id = int(row['hesc_id']) # 'hesc_id' on XuTestHEA() entries!
                hsd_idx = hsd.get_idx_by_hesc_id(hesc_id)

                group_idx_internal = idx2group_internal[hsd_idx]
                group_idx_to_xu49 = idx2group_to_xu49[hsd_idx]
                group_entries_internal = [int(hsd[j]['hesc_id']) for j in dupl_group_internal[group_idx_internal]]
                if idx in indices_not_found:
                    assert group_idx_to_xu49 is None
                    warnings.warn(f'xu_dataset_idx:{idx} does not grouped with any entries on the HE-SC dataset, directly using hesc_id',UserWarning)
                    similar_entries_for_df.append(str(group_entries_internal))
                else:
                    group_entries_to_xu49 = [int(hsd[j]['hesc_id']) for j in dupl_group_to_xu49[group_idx_to_xu49]]
                    assert group_entries_to_xu49==group_entries_internal
                    similar_entries_for_df.append(str(group_entries_to_xu49))
        print(indices_not_found)
        print([xu_test_hea[i] for i in indices_not_found])
        if indices_not_found!=[38, 39]:
            warnings.warn(f"indices_not_found={indices_not_found}",TestSnapshotWarning)
        else:
            print(f'ss_dataset._df.shape: {ss_dataset._df.shape}')

    def test_end_user(self):
        """
        Docstring for test_end_user; temp for debug.
        
        :param self: Description
        """
        with get_package_dataset() as path:
            test_dataset_path = path
        print(D2TableDataset(test_dataset_path, exception_col=None)._df.shape)
   
#if __name__=="__main__":
    #unittest.main()
    #TestDataset("test_end_user").debug()
    #TestDataset("test_dataset").debug()
    #TestDataset("test_comparison").debug()