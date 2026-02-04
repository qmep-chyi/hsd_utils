"""package's optional test module

left snapshot, featurized_snapshot_2025-11-25.csv

main features:
    * 
    $ 
"""

import unittest
import importlib.resources as resources
from pathlib import Path

from hsdu.dataset import D2TableDataset
from hsdu.convert.utils import Converter 
from hsdu.convert.feature import  MultiSourceFeaturizer

class Test(unittest.TestCase):
    """Test core class methods"""

    def test_convert(self):
        """in-house dataset test
        
        load the in-house dataset
        """
        test_dataset_path: Path=None
        with resources.as_file(resources.files("hsdu.data.tests") /"full_dataset.csv") as path:
            test_dataset_path = path if Path(path).is_file() else None

        if test_dataset_path is None:
            with resources.as_file(resources.files("hsdu.data.tests") /"test_dataset.csv") as path:
                test_dataset_path = path

        # generate cleaned datatable with comositional columns only
        converter = Converter(test_dataset_path, "compositional5_merge2maxTc_wholedataset.json")
        converter.convert(make_dir=True, exist_ok=True)

        # featurize from cleand datatable
        dataset = D2TableDataset(converter.save_compositional5_pth, exception_col=None)
        dataset.idx2aux['comps_pymatgen']

        featurizer=MultiSourceFeaturizer(config="xu.json")
        featurized_df = featurizer.featurize_all(dataset, merge_both=True, save_file="test_featurized_table.csv")
        print(featurized_df)

        # delete temporal files
        #converter.save_compositional5_pth.unlink()
        #converter.save_log_pth.unlink()
        #Path("test_featurized_table.csv").unlink()
        

    
if __name__ == '__main__':
    Test("test_convert").debug()