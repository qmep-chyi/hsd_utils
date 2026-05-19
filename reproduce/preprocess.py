"""
# preprocess, from raw dataset to featurized table (used as model inputs)
- preprocess and clean datatable
- featurization

# Config files: 
- see `hsd_utils\src\hsdu\config\` directory

## preprocess (clean) configs
* maxTc.json
    ```json
    ...
    "duplicates_rule":{
        "tc": {
            "order": "highest",
            "sort_by": "max_Tc"
        },
        ...
    ```
    * leave a single entry that had highest Tc.
* all_tcs_test.json
    ```json
    ...
    "duplicates_rule":{
        "tc": {
            "order": "all_tcs",
            "sort_by": "max_Tc"
        },
        ...
    ```
    * re-derive Tc stats all over the valid Tc values.
* supercon.json

## Featurize config
* comp450.json
* comp146.json

## Dataset config

"""
#%%

#%% 
# 
import pandas as pd
from matminer.featurizers.base import MultipleFeaturizer

from hsdu.preprocess.utils import featurizer_config_loader
from hsdu.dataset import D2TableDataset, Dataset
from hsdu.preprocess.utils import Preprocessor 

def preprocess(raw_dataset_path, preprocess_config):
    # generate cleaned datatable with comositional columns only
    hsd = Dataset(raw_dataset_path, exception_col='Exceptions')
    preprocessor = Preprocessor(hsd, preprocess_config)
    return preprocessor.convert()

def featurization(cleaned_df, featurizer_config):
    #featurization
    dataset = D2TableDataset(cleaned_df, exception_col=None)
    featurized_df = pd.DataFrame()
    featurized_df['comps_pymatgen']=dataset.idx2aux['comps_pymatgen']

    featurizers_list, col_names_df = featurizer_config_loader(config=featurizer_config)
    featurizer = MultipleFeaturizer(featurizers_list)
    featurizer.set_n_jobs(1) #TODO: if not, MultipleFeaturizer raise errors on windows11
    featurizer.featurize_dataframe(featurized_df, col_id='comps_pymatgen', inplace=True)

    # comps_pymatgen column is a Composition object so drop or get string.
    featurized_df['comps_pymatgen']=featurized_df['comps_pymatgen'].apply(lambda x:x.to_pretty_string())
    return featurized_df, col_names_df


preprocess_config_list = ['maxTc']
featurize_config_list = ['comp146', 'comp450']

# your path to the dataset
dataset_path = r'$(path_to_hsdu_pacakge)\src\hsdu\data\dataset_20260511.csv'

for pconfig in preprocess_config_list:

    cleaned_df = preprocess(dataset_path, pconfig)
    for fconfig in featurize_config_list:
        featurized_df, col_names_df = featurization(cleaned_df, fconfig)
        assert len(cleaned_df)==len(featurized_df)
        out_df = pd.concat([cleaned_df.reset_index(drop=True), featurized_df.reset_index(drop=True)], axis=1)
        out_df.to_csv(f'{pconfig}_{fconfig}.csv')
        col_names_df.to_json(f'{pconfig}_{fconfig}_features_list.json', orient='index', indent=4)