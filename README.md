# Utils for HS dataset
***Repository for collaboration, not for public share***

## Changes
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
    `uv run python -m unittest test`
    `uv run python -m unittest test_convert`

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