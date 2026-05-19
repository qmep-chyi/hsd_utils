"""package's optional test module

main features:
    * clean, preprocess raw datatable
    * featurization
"""

import unittest
import warnings

from matminer.featurizers.base import MultipleFeaturizer
import pandas as pd

from hsdu.dataset import Dataset, D2TableDataset
from hsdu.preprocess.utils import Preprocessor 
from hsdu.preprocess.feature import featurizer_config_loader
from hsdu.utils.utils import config_parser, feature_col_name_parser
from .utils_for_test import TestSnapshotWarning, get_package_dataset

class Test(unittest.TestCase):
    """Test core class methods"""

    def test_preprocess(self):
        """in-house dataset test
        
        load the in-house dataset
        """
        with get_package_dataset(version='20260511') as path:
            test_dataset_path = path

        # generate cleaned datatable with comositional columns only
        hsd = Dataset(test_dataset_path, exception_col='Exceptions')
        converter = Preprocessor(hsd, "maxTc.json")
        cleaned_table = converter.convert(make_dir=True, exist_ok=True)

        # featurize from cleand datatable
        dataset = D2TableDataset(cleaned_table, exception_col=None)
        featurized_df = pd.DataFrame()
        featurized_df['comps_pymatgen']=dataset.idx2aux['comps_pymatgen']

        featurizers_list, col_names_df = featurizer_config_loader(config='comp450.json')
        featurizer = MultipleFeaturizer(featurizers_list)
        featurizer.set_n_jobs(1)
        featurizer.featurize_dataframe(featurized_df, col_id='comps_pymatgen', inplace=True)
        featurized_df['comps_pymatgen']=featurized_df['comps_pymatgen'].apply(lambda x:x.to_pretty_string())

        featurized_df.to_csv('test_featurized_table.csv')
        print(featurized_df)

        # delete temporal files
        #converter.save_compositional5_pth.unlink()
        #converter.save_log_pth.unlink()
        #Path("test_featurized_table.csv").unlink()
    def test_preprocess_all_tcs(self):
        # featurize with `all_tc_test.json` config, where config['duplicates_rule']['tc']['order']=="all_tcs"
        with get_package_dataset(version='20260511') as path:
            test_dataset_path = path

        # generate cleaned datatable with comositional columns only
        hsd = Dataset(test_dataset_path, exception_col='Exceptions')
        converter = Preprocessor(hsd, "all_tcs_test.json")
        cleaned_table = converter.convert(make_dir=True, exist_ok=True)
        dataset = D2TableDataset(cleaned_table, exception_col=None)
        dataset.idx2aux['comps_pymatgen']

        featurized_df = pd.DataFrame()
        featurized_df['comps_pymatgen']=dataset.idx2aux['comps_pymatgen']

        featurizer_list, _ =featurizer_config_loader(config="comp450.json")
        featurizer = MultipleFeaturizer(featurizer_list)
        featurizer.set_n_jobs(1) #TODO: raise error without this.. maybe from InhouseSecondary?
        featurizer.featurize_dataframe(featurized_df, col_id='comps_pymatgen', inplace=True)
        if featurized_df.shape!=(256, 511):
            warnings.warn(f"featurized_df.shape={featurized_df.shape}",TestSnapshotWarning)
        else:
            print(f'featurized_df.shape: {featurized_df.shape}')
        featurized_df['comps_pymatgen']=featurized_df['comps_pymatgen'].apply(lambda x:x.to_pretty_string())
        featurized_df.to_csv("test_featurized_table_all_tcs.csv")

    def test_feature_col_utils(self):
        default_config = config_parser('comp450', mode='featurize')
        _, feature_cols_df = featurizer_config_loader(default_config)
        if len(feature_cols_df)!=450:
            warnings.warn(len(feature_cols_df), TestSnapshotWarning)

#if __name__ == '__main__':
    #Test('test_feature_col_utils').debug()
    #Test('test_convert_all_tcs').debug()
    #Test("test_convert").debug()