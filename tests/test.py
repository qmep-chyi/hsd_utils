"""package's test module

main features:
    * load test version of in-house dataset
    * load additional dataset from literatures.
        * StanevSuperCon
        * XuDataset (see comparison.XuDataset for details)

    $ python -m unittest test.py
"""

import unittest
import importlib.resources as resources

import pandas as pd
from sklearn.metrics import r2_score

from hsdu.dataset import Dataset, D2TableDataset
from hsdu.comparison import XuTestHEA, StanevSuperCon

class Test(unittest.TestCase):
    """Test core class methods"""

    def test_dataset(self):
        """in-house dataset test
        
        load test version of in-house dataset
        """
        # load dataset
        with resources.as_file(resources.files("hsdu.data.tests") /"test_dataset.csv") as path:
            test_dataset_path = path
        dataset = Dataset(test_dataset_path, config="default.json")

        # load stanev's supercon dataset 
        ss_dataset = StanevSuperCon()
        ss_dataset.pymatgen_comps()

        xu_test_set = XuTestHEA()

        # load as a D2TableDataset
        dataset = D2TableDataset(test_dataset_path, exception_col=None)
        dataset.pymatgen_comps()

        #xu_val_r2score: test XuDataset reproduces the snapshot
        xu_dataset = XuTestHEA()
        r2score = r2_score(xu_dataset.dataframe["Experimental_T_c(K)"], xu_dataset.dataframe["Predicted_T_c(K)"])
        print(f"r2_score:{r2score}")
        self.assertAlmostEqual(r2score, 0.9246, places=4)
    
    def test_notebook(self):
        # trying to keep compatibility with 
        # keep compatibility with https://colab.research.google.com/drive/1SUAELDp_BLy62RK5ilqmMbG0QMEvP-J3
        
        dataset = Dataset("/content/sample_data/inhouse_dataset_merged_0810.xlsx", config="default.json")

        # dataset.dataframe is pandas DataFrame
        assert isinstance(dataset.dataframe, pd.DataFrame)
        
if __name__ == '__main__':
    Test().test_dataset() #it is convenience for debug..
    #unittest.main()