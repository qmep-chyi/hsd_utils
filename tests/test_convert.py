"""package's optional test module

left snapshot, featurized_snapshot_2025-11-25.csv

main features:
    * 



    $ python -m unittest test.py
"""

import unittest
import importlib.resources as resources
from pathlib import Path

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
        
        # generate cleaned datatable
        from hsdu.convert.utils import Converter 
        converter = Converter(test_dataset_path, "compositional5_merge2maxTc_wholedataset.json")
        converter.convert(make_dir=True, exist_ok=True)

        # featurize from cleand datatable
        from hsdu.dataset import D2TableDataset
        from hsdu.convert.feature import  MultiSourceFeaturizer
        dataset = D2TableDataset(converter.save_compositional5_pth, exception_col=None)
        dataset.pymatgen_comps()

        featurizer=MultiSourceFeaturizer(config="xu.json")
        featurized_df = featurizer.featurize_all(dataset.df, merge_both=True, save_file="test_featurized_table.csv")

        # delete temporal files
        converter.save_compositional5_pth.unlink()
        converter.save_log_pth.unlink()
        Path("test_featurized_table.csv").unlink()
        

    
if __name__ == '__main__':
    #Test().test_dataset() #it is convenience for debug..
    #unittest.main()
        
    # old test codes; to be refactored
    TEMP_TEST_FEATURIZE=True
    TEMP_TEST_COMPARISON_TABLES=False

    
    if TEMP_TEST_FEATURIZE:
        from pathlib import Path

        import pandas as pd

        from hsdu.dataset import Dataset, D2TableDataset

        # load dataset
        merged_dataset_path=Path(r"C:\Users\chyi\hsd_utils\temp_devs\hesc_dataset_16dec2025 - DataTable.csv")

        # generate cleaned datatable
        from hsdu.convert.utils import Converter 
        converter = Converter(merged_dataset_path, "compositional5_merge2maxTc_wholedataset.json")
        converter.convert(make_dir=True, exist_ok=True)

        # featurize from cleand datatable
        from hsdu.dataset import D2TableDataset
        from hsdu.convert.feature import  MultiSourceFeaturizer
        dataset = D2TableDataset(converter.save_compositional5_pth, exception_col=None)
        dataset.pymatgen_comps()

        featurizer=MultiSourceFeaturizer(config="xu.json")
        featurized_df = featurizer.featurize_all(dataset.df, merge_both=True, save_file="featurized_table_16dec2025.csv")
    if TEMP_TEST_COMPARISON_TABLES:
        from hsdu.comparison import XuTestHEA, StanevSuperCon
        from hsdu.convert.utils import Converter 
        ss_dataset = StanevSuperCon(config="ss.json")
        converter = Converter(ss_dataset, "comp5_ss.json")
        converter.convert(make_dir=True, exist_ok=True, simple_target=True)

        #%%
        from hsdu.dataset import D2TableDataset
        from hsdu.convert.feature import  MultiSourceFeaturizer
        dataset = D2TableDataset(converter.save_compositional5_pth, exception_col=None)
        dataset.pymatgen_comps()

        featurizer=MultiSourceFeaturizer(config="xu.json")
        featurized_df = featurizer.featurize_all(dataset.df, merge_both=True, save_file="test_featurized_table.csv")