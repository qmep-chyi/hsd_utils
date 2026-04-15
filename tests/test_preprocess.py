"""package's optional test module

main features:
    * 
    $ 
"""

import unittest, warnings

from hsdu.dataset import Dataset, D2TableDataset
from hsdu.convert.utils import Preprocessor 
from hsdu.convert.feature import MultiSourceFeaturizer
from hsdu.utils.utils import config_parser, init_feature_config
from .utils_for_test import ConsistentResultsError, get_package_dataset

class Test(unittest.TestCase):
    """Test core class methods"""

    def test_preprocess(self):
        """in-house dataset test
        
        load the in-house dataset
        """
        with get_package_dataset() as path:
            test_dataset_path = path

        # generate cleaned datatable with comositional columns only
        hsd = Dataset(test_dataset_path, exception_col='Exceptions')
        converter = Preprocessor(hsd, "compositional5_merge2maxTc_wholedataset.json")
        converter.convert(make_dir=True, exist_ok=True)

        # featurize from cleand datatable
        dataset = D2TableDataset(converter.save_compositional5_pth, exception_col=None)
        dataset.idx2aux['comps_pymatgen']

        featurizer=MultiSourceFeaturizer(config="comp450.json")
        featurized_df = featurizer.featurize_all(dataset, merge_both=True, save_file="test_featurized_table.csv")
        print(featurized_df)

        # delete temporal files
        #converter.save_compositional5_pth.unlink()
        #converter.save_log_pth.unlink()
        #Path("test_featurized_table.csv").unlink()
    def test_preprocess_all_tcs(self):
        # featurize with `all_tc_test.json` config, where config['duplicates_rule']['tc']['order']=="all_tcs"
        with get_package_dataset() as path:
            test_dataset_path = path

        # generate cleaned datatable with comositional columns only
        hsd = Dataset(test_dataset_path, exception_col='Exceptions')
        converter = Preprocessor(hsd, "all_tcs_test.json")
        converter.convert(make_dir=True, exist_ok=True)
        dataset = D2TableDataset(converter.save_compositional5_pth, exception_col=None)
        dataset.idx2aux['comps_pymatgen']

        featurizer=MultiSourceFeaturizer(config="comp450.json")
        featurized_df = featurizer.featurize_all(dataset, merge_both=True, save_file="test_featurized_table.csv")
        if featurized_df.shape!=(255, 511):
            warnings.warn(f"featurized_df.shape={featurized_df.shape}",ConsistentResultsError)
        else:
            print(f'featurized_df.shape: {featurized_df.shape}')

    def test_feature_col_utils(self):
        default_config = config_parser('comp450', mode='featurize')
        n_cols, col_names, feature_cols_df = init_feature_config(default_config)
        if len(feature_cols_df)!=450:
            warnings.warn(len(feature_cols_df), ConsistentResultsError)
        if n_cols!=450:
            warnings.warn(n_cols, ConsistentResultsError)
        if len(col_names)!=450:
            warnings.warn(len(col_names), ConsistentResultsError)

#if __name__ == '__main__':
    #Test('test_feature_col_utils').debug()
    #Test('test_convert_all_tcs').debug()
    #Test("test_convert").debug()