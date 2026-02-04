"""package's test module

main features:
    * load test version of in-house dataset
    * load additional dataset from literatures.
        * StanevSuperCon
        * XuDataset (see comparison.XuDataset for details)

    $ uv run python -m unittest test
    
"""

import importlib.resources as resources
import unittest
from pathlib import Path

from sklearn.metrics import r2_score

from hsdu.dataset import Dataset, D2TableDataset
from hsdu.comparison import XuTestHEA, StanevSuperCon

class TestDataset(unittest.TestCase):
    def test_dataset(self):
        """in-house dataset test
        
        if full_dataset.csv exists in hsdu.data.tests, use that. except, use test_dataset.csv
        """
        test_dataset_path: Path=None
        with resources.as_file(resources.files("hsdu.data.tests") /"full_dataset.csv") as path:
            test_dataset_path = path if Path(path).is_file() else None

        if test_dataset_path is None:
            with resources.as_file(resources.files("hsdu.data.tests") /"test_dataset.csv") as path:
                test_dataset_path = path

        # load as a D2TableDataset
        dataset = D2TableDataset(test_dataset_path, exception_col=None)
        print(dataset[3])

        # load dataset
        dataset = Dataset(test_dataset_path, config="default.json")
        
    def test_comparison(self):
        # load stanev's supercon dataset 
        ss_dataset = StanevSuperCon()
        ss_dataset.idx2aux['comps_pymatgen']

        _ = XuTestHEA()

        #xu_val_r2score: test XuDataset reproduces the snapshot
        xu_dataset = XuTestHEA()
        r2score = r2_score(xu_dataset._df["Experimental_T_c(K)"], xu_dataset._df["Predicted_T_c(K)"])
        print(f"r2_score:{r2score}")
        self.assertAlmostEqual(r2score, 0.9246, places=4)
    
    def test_end_user(self):
        """
        Docstring for test_end_user; temp for debug.
        
        :param self: Description
        """
        _ = D2TableDataset(r"C:\Users\chyi\hsd_utils\tests\temp_devs\HESC251.csv", exception_col=None)
        
   
if __name__=="__main__":
    #unittest.main()
    TestDataset("test_end_user").debug()
    TestDataset("test_dataset").debug()
    TestDataset("test_comparison").debug()