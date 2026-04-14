# Utils for HS dataset
***Repository for collaboration, not for public share***

## Changes
* (2026-04-14) updates utility attributes on `MultiSourceFeaturizer` from `hsdu.convert.feature`
    * renamed 'source' key of featurize config to 'featurizer'
        * it decides featurization method 
            * in between `MultiSourceFeaturizer().featurize_matminer()` and `MultiSourceFeaturizer().featurize_matminer2nd()`
            * on `MultiSourceFeaturizer().init_feature_config()` from `hsdu.convert.feature`
        * valid values are: `["matminer", "matminer_expanded", "matminer_secondary"]`
        * Example usage in repository: `config_parser('xu',mode='featurize')` from `hsdu.utils.utils`
    * `MultiSourceFeaturizer().col_names_df`
        * refactored `MultiSourceFeaturizer().init_feature_config()`
            * to `init_feature_config()` from `hsdu.utils.utils`
            * now it returns (feature_count, col_names:list, col_names_df)
        * `col_names_df` (pd.DataFrame):
            * dataframe with `columns=['col_name', 'feature','stat','weigthed', 'source', 'featurizer']`
            * criteria from config json file, for `feature selection` or `feature importance`
    * new convert test (see `tests/test_conver.py`, `test_convert_all_tcs()`) with `hsdu\config\convert\all_tcs_test.json`
        * in this case, it re-derives max, min, average over the whole valid tc values to be merged.
        * config on TcMerger() from `hsdu.convert.utils`, `self.rule['order']=="all_tcs":`
        * manually checked a case: `duplicated_comps[0]=[15, 277, 322, 323]`
            * valid tcs: [7.12, 6.78, 7.45, 6.69, 6.42, 6.96, 8.43, 7.56, 7.2]
            * but fraction of `10_277` was wrong (updated 'changes' on the shraed project drive)
    * All tests passed. Tagged as 'v0.1.2'

* (2026-04-10)
    * ***unintended merging duplicates rule***: 
        * with default config, `avg_Tc` and `min_Tc` was not re-derived from all the valid T_c but from the entry which had `max_Tc`.
        * updated config file with , `src\hsdu\config\convert\compositional5_merge2maxTc_wholedataset.json`.
            * previously, `"duplicates_rule":{"tc":"max_Tc"}`. it meant A. merge to highest T_c, B. on the `max_Tc` column.
            * now it is separated.
                ```json
                "tc": {
                    "order": "highest",
                    "sort_by": "max_Tc"
                }
                ```
        * codes updates upon these changes
    * removed outdated config files, updated convert config json files.
    * merge duplicates log now shows 
* (2026-04-03) Refactored `tests/temp_new_group_algorithm.py` and other temporal codes in `tests/`
    * replaced old duplicate group method with new one
        - old: hsdu.data.Dataset().pymatgen_duplicates(rtol=0.02)
    * Removed temporal/dev scripts and refactored as `tests/test_group_duplicates.py`
    * All test (`test`, `test_group_duplicates`, `test_convert`) passed with latest HE-SC dataset `merged_dataset_20260403` (on google drive)
* (03 Feb 2026) New Features / functions
    * New attribute `Dataset.idx2aux`-- dict of mappings: `aux_name -> (idx -> value)`.
        * e.g. `Dataset.idx2aux["comps_pymatgen"][3]` is pymatgen Composition of entry idx=3.
        * Now, do not add new column on Dataset().df or D2TableDataset().df
        * so, `dataset.df["comps_pymatgen"]` removed. use `dataset.idx2aux["comps_pymatgen"]`
        * make sure indices are `list(range(len(dataset)))`
            ```python
            assert dataset.df.index.tolist()==list(range(len(dataset)))
            assert dataset.df.index.tolist()==dataset.idx2aux["aux_name"].keys()
            ```
        * Now, `dataset[idx]` returns a dictionary;
            * `out = {k:v[idx] for k, v in self.idx2aux.items()}`
            * `out["df"] = self.df.loc[idx, :]`
    * new distance matrix, duplicates checker + merger
        * using `scipy.spatial.distance.cdist`
        * See `hsdu.utils.duplicate`
        * Now distances and grouping results are order-independent. (previously it was not.)
    * One-hot vector like encoding on initiallization
        * integrated initialization steps
        * now it adds columns like `Ta`, `Hf` for self.df (iupac ordered)
        * values are fraction of that elements
            * Not normalized (e.g. `sum(self.df.loc[idx, [self.elemental_set]])!=1`)
            * refers to `one-hot fracs` from now on.
        
* (19 Jan 2026) Now using `uv` without `Poetry`
    * optional dependency `convert` for featurization and convert.
    * with `convert` group installed, `featurizer` and `converters` will be available

## Installation
### Virtual Environment via UV (Recommended)
devloped environment: windows 11, python==3.13, uv==0.9.24
* install `git clone https://github.com/qmep-chyi/hsd_utils.git`
* `cd hsd_utils`
* `uv sync`
* to use featurizer/converter, `uv sync --dev --extra convert` 
    * As [`matminer` may not support python>3.13](https://github.com/hackingmaterials/matminer/blob/main/pyproject.toml) and many latest envs.
* (recommended) test (change directory to `tests/`)
    * `uv run python -m unittest test`
    * `uv run python -m unittest test_convert`
    * `uv run python -m unittest test_group_duplicates`
    * if required, put full dataset as `hsd_utils/src/hsdu/data/tests/full_dataset.csv`


### Google colab(linux) and pip
***To be tested.(leave versions python and major package when test)***
* `!git clone https://github.com/qmep-chyi/hsd_utils.git`
* `!pip install -e hsd_utils/` ***should success***
    * may  not install `convert` group.


## Usage
### Load in-house dataset
- Load latest in-house dataset as a `pandas.DataFrame`,
- get featurized dataset as pandas.DataFrame.  
- *dataset is not here. download from (private) google drive.*

```python
from hsdu.dataset import Dataset
import pandas as pd

hsd = Dataset(your_dataset_path, config="default.json") 
# hsd.df is pandas DataFrame
assert isinstance(hsd.df, pd.DataFrame)
```

### Convert tables and Featurize
***Should install optional(`convert`) requirements, see `Installation`***

* get cleaned datatable to generate compositional feature
    * clean some inhomogeneous entries
    * merge multiple $T_c$ values from multiple measurement
    * merge entries with too close composition
* then featurize, 
    * get datatable with massive features(450 columns for input.)
```python
import importlib.resources as resources
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score

from hsdu.dataset import Dataset, D2TableDataset

# load dataset
merged_dataset_path="some_path"

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
featurized_df = featurizer.featurize_all(dataset.df, merge_both=True, save_file="test_featurized_table.csv")
```

### Load datatables of previous study for comparison
***not tested after re-init***
* `hsdu.comparison.XuTestHEA()`: the validation data [Xu 2025(paper)](https://journal.hep.com.cn/fop/EN/10.15302/frontphys.2025.014205) employed, in supplements.
* `hsdu.comparison.StanevSuperCon()`: Preprocessed [Supercon data](https://github.com/vstanev1/Supercon/blob/master/Supercon_data.csv), like train set of Xu 2025. See [preprocess_supercon.md](src\hsdu\data\miscs\preprocess_supercon.md) for details.
* `hsdu.comparison.KitagawaTables()`: Not Implemented yet
* `hsdu.comparison.CayadoTables()`: Not Implemented yet
```python
from sklearn.metrics import r2_score
import pandas as pd

from hsdu.comparison import XuTestHEA, StanevSuperCon

xu_dataset = XuTestHEA()
print(r2_score(xu_dataset.df["Experimental_T_c(K)"], xu_dataset.df["Predicted_T_c(K)"]))

ss_dataset = StanevSuperCon(config="ss.json")
assert isinstance(ss_dataset.df, pd.DataFrame)
print(isinstance(ss_dataset.df.loc[0, "elements_fraction"][0], float))
```