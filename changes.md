
# Changes
## v0.1.3(2026-04-15) remove outdated things, renamed some names with 'xu'
    * fixed wrong indentation
        * added tests about this, see `test_feature_col_utils()` on `tests/test_convert.py`
    * removed outdated codes and files
    * changed some names including 'xu` as they are not exactly reproducing what Xu et al. (2025) did.
        * on `src\hsdu\data\miscs\xu2025_test_HEAs.csv`
            * removed `xu_index` and `Predicted_T_c(K)` 
            * because of their license is not clear
            * And sorted by the index of Kitagawa et al. (2020)
            * So, let's say they are subset of Kitagawa et al. (2020).
    * now all the tests requires **full_dataset.csv** available at project-team shared drive (not public yet!)
        * should exists: `src\hsdu\data\tests\test_dataset.csv`
        * to reduce maintenace workload
    * `Converter` refactored to `Converter` and `Preprocessor`
        * `Converter` for common convert tables work
        * `Preprocessor` for preprocess raw HE-SC dataset
    * add some tests about v0.1.2 and v0.1.3
## v0.1.2(2026-04-14) updates utility attributes on `MultiSourceFeaturizer` from `hsdu.convert.feature`
    * renamed 'source' key of featurize config to 'featurizer'
        * it decides featurization method 
            * in between `MultiSourceFeaturizer().featurize_matminer()` and `MultiSourceFeaturizer().featurize_matminer2nd()`
            * on `MultiSourceFeaturizer().init_feature_config()` from `hsdu.convert.feature`
        * valid values are: `["matminer", "matminer_expanded", "matminer_secondary"]`
        * Example usage in repository: `config_parser('comp450',mode='featurize')` from `hsdu.utils.utils`
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
## (2026-04-10)
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
## (2026-04-03) Refactored `tests/temp_new_group_algorithm.py` and other temporal codes in `tests/`
    * replaced old duplicate group method with new one
        - old: hsdu.data.Dataset().pymatgen_duplicates(rtol=0.02)
    * Removed temporal/dev scripts and refactored as `tests/test_group_duplicates.py`
    * All test (`test`, `test_group_duplicates`, `test_convert`) passed with latest HE-SC dataset `merged_dataset_20260403` (on google drive)
## (03 Feb 2026) New Features / functions
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
        
## (19 Jan 2026) Now using `uv` without `Poetry`
    * optional dependency `convert` for featurization and convert.
    * with `convert` group installed, `featurizer` and `converters` will be available
