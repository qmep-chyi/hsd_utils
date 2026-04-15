# Utils for HS dataset
***Repository for collaboration, not for public share***

## Recent Changes
* `v0.1.3` (2026-04-15) 
    * fixed wrong indentation
    * removed outdated things
    * changed some names including 'xu` as they are not exactly reproducing what Xu et al. (2025) did
    * now all the tests requires **full_dataset.csv** available at project-team shared drive (not public yet!)
    * renamed `Converter` to `Preprocessor`
* `v0.1.2` (2026-04-14) updates utility attributes on `MultiSourceFeaturizer` from `hsdu.convert.feature`
    * renamed 'source' key of featurize config to 'featurizer'
    * `MultiSourceFeaturizer().col_names_df`
    * new convert test (see `tests/test_conver.py`, `test_convert_all_tcs()`) with `hsdu\config\convert\all_tcs_test.json`
    * All tests passed. Tagged as 'v0.1.2'
* See [changes.md](changes.md) for detail

## Kind of Dataset (Terms):
* `Raw dataset`: raw HE-SC dataset with various $T_c$ values and additional informations
* `preprocessed table`: cleaned, preprocessed datatable. `convert`ed from the Raw dataset.
    * merging duplicates have close compositions
    * drop invalid entries
    * process target: merge $T_c$s 
    * default config: `src\hsdu\config\convert\compositional5_merge2maxTc_wholedataset.json`
* `featurized table`: includes `features` columns that will be ML model inputs.
    * default 450 compositional features.
    * featurized from `preprocessed table`
    * default config: `src\hsdu\config\featurize\comp450.json`
    * the term `featurizer` came from [pymatgen](https://github.com/hackingmaterials/matminer)

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
        * outputs on v0.1.3 (2026-04-15), with full dataset
    * `uv run python -m unittest test_group_duplicates`
    * if required, put full dataset as `hsd_utils/src/hsdu/data/tests/full_dataset.csv`


### Google colab(linux) and pip
***To be tested.(leave versions python and major package when test)***
* `!git clone https://github.com/qmep-chyi/hsd_utils.git`
* `!pip install -e hsd_utils/` ***should success***
    * may  not install `convert` group.


## Usage
* Temporally, please see tests
    * tests/test.py