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
from .utils_for_test import ConsistentResultsError, get_package_dataset


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        """in-house dataset test
        
        full_dataset.csv sholud exists in hsdu.data.tests
        """
        with get_package_dataset() as path:
            test_dataset_path = path
        # load as a D2TableDataset
        dataset = D2TableDataset(test_dataset_path, exception_col=None)

        # load dataset
        dataset = Dataset(test_dataset_path, config="default.json")
        if dataset._df.shape!=(432, 105):
            warnings.warn(f"ss_dataset._df.shape={dataset._df.shape}",ConsistentResultsError)
        else:
            print(f'dataset._df.shape: {dataset._df.shape}')
    def test_comparison(self):
        # load stanev's supercon dataset 
        ss_dataset = StanevSuperCon()
        ss_dataset.idx2aux['comps_pymatgen']
        #print(f"First entry of supercon(stanev version): {ss_dataset[0]}")
        
        if ss_dataset._df.shape!=(6194, 86):
            warnings.warn(f"ss_dataset._df.shape={ss_dataset._df.shape}",ConsistentResultsError)
        else:
            print(f'ss_dataset._df.shape: {ss_dataset._df.shape}')

        xu_test45 = XuTestHEA()
        if xu_test45._df.shape!=(45, 26):
            warnings.warn(f"xu_test45._df.shape={xu_test45._df.shape}",ConsistentResultsError)
        else:
            print(f'xu_test45._df.shape: {xu_test45._df.shape}')
        #print(f"First entry of xu2025_test_hea: {xu_test45[0]}")
    
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