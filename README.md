# Utils for HS dataset
***Repository for collaboration, not for public share yet***
## Notes
### 24 Nov. Reverting abandoned features
* (trying) optional installation via `poetry install --with convert` or `poetry sync --with convert`
    * default install keep minimal utilities for the dataset
    * with `convert` group installed, `featurizer` and `converters` will be available
* while [`matminer` may not support python>3.13](https://github.com/hackingmaterials/matminer/blob/main/pyproject.toml) and many latest envs, it shoud be separated from.
* then it is just refactored and re-initialized repo from draftsh2025.
### Init. (10 Nov 2025)
* refactoring, split from [draftsh2025, `2973dea`](https://github.com/chhyyi/draftsh2025/commit/2973dea7f3fc0e86efe12acd94cf6cd09a438261)
* without featurization, datatable conversion


## Installation
### `Poetry` Virtual Environment (Recommended)
devloped environment: windows 11, python==3.13, poetry version 2.1.4
* install `git clone https://github.com/qmep-chyi/hsd_utils.git`
* `cd hsd_utils`
* additional poetry config may be required: `poetry config virtualenvs.use-poetry-python true`
* `poetry install --with dev` shoulde success, include `Installing the current project: hsdu`
    * `poetry install --with dev,convert` to use featurizer/converter.
* restart terminal, `cd` to repo root.
* `poetry env list` to assure you are running on the virtual environment.
* select python interpreter on the IDLE if required

### Google colab(linux) and pip
***To be tested.(leave versions python and major package when test)***
* `!git clone https://github.com/qmep-chyi/hsd_utils.git`
* `!pip install -e hsd_utils/` ***should success***
    * may  not install `convert` group.


## Usage
### Load in-house dataset
Load latest in-house dataset as a `pandas.DataFrame`,
get featurized dataset as pandas.DataFrame.  
*dataset is not here. download at (private) google drive.*

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
    * merge multiple $T_c$s from multiple measurement
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