"""utility functions 

functions:
    * config_parser()
"""

import json
import math
from pathlib import Path
from typing import Literal, Iterator
import importlib.resources as resources
from itertools import product

import pandas as pd
import numpy as np

def config_parser(config: str | dict | Path, mode: Literal["dataset", "featurize", "convert"]):
    """
    load and parse config
    """
    if isinstance(config, str):
        config = Path(config)
    else:
        pass

    if isinstance(config, Path):
        if config.is_absolute():
            xls_path = config
        else:
            with resources.as_file(resources.files("hsdu.config").joinpath(mode).joinpath(config.name)) as path:
                xls_path = path
                xls_path = xls_path.with_suffix(".json")
                assert xls_path.is_file(), FileNotFoundError(xls_path)
        fp = open(xls_path, "r", encoding = "utf-8")
        assert Path.is_file(xls_path)
        config = json.load(fp)
        fp.close()
    
    assert isinstance(config, dict), f"config: {config}"
    return config

def init_feature_config(config:dict):
    features = []
    statistics=[]
    sources=[]
    weigthed=[]
    featurizers=[]
    
    num_features = 0
    col_names = []
    for featurizer in config["featurizers"]:
        if featurizer in ("matminer", "matminer_expanded", "matminer_secondary"):
            assert isinstance(config[featurizer], list), f"config[source] should be a list of dictionaries but: {config[featurizer]}"
            for config_1source in config[featurizer]:
                config_single_source = ConfigSingleSource(config_1source)
                num_features += len(config_single_source)
                for srcc, feat, stat in config_single_source.iter_config():
                    #delete some parameters used when featurize. see `hsdu\config\feature\xu.json`
                    col_name=f'{srcc}_{feat}_{stat.replace("::","_")}'
                    col_name=col_name.replace("_self_prop::", "_")
                    col_name=col_name.replace("_self_prop", "")
                    col_names.append(col_name)
                    features.append(feat)
                    if '::' in stat:
                        parsed_stat = stat.split('::')
                        assert len(parsed_stat)<=2
                        if parsed_stat[0]=='self_prop':
                            statistics.append(parsed_stat[1])
                            weigthed.append(None)
                        else:
                            statistics.append(parsed_stat[0])
                            weigthed.append(parsed_stat[1])
                    else:
                        statistics.append(stat if stat!='self_prop' else None)
                        weigthed.append(None)
                    sources.append(srcc)
                    featurizers.append(featurizer)
        elif featurizer == "materials_project":
            raise NotImplementedError(featurizer)
        else:
            raise ValueError(featurizer)
    col_names_df = pd.DataFrame(zip(col_names, features, statistics, weigthed, sources, featurizers), columns=['col_name', 'feature','stat','weigthed', 'source', 'featurizer'])

    return num_features, col_names, col_names_df
    
def merge_dfs(src_df: pd.DataFrame, featurized_df: pd.DataFrame, reset_index=True):
    """
    merge dfs with same length"""
    assert reset_index, NotImplementedError(reset_index)

    shape_df = np.shape(featurized_df)
    shape_src_df = np.shape(src_df)
    featurized_df = src_df.reset_index(drop=True).join(featurized_df.reset_index(drop=True))

    assert featurized_df.shape[1] == shape_df[1]+shape_src_df[1]
    return featurized_df

class ConfigSingleSource():
    """For a single run of featurize_dataframe() of `draftsh.feature.MyElementalProperty()` class.

    args:
        * config_dic: dict
        * iter_vars: list[str]
            * config_dic[iter_vars[idx]]: list[str] for valid idx.
            * should follow the loop - hierarchy of featurizer like `matminer.featurizers.composition.composite.Elementproperty().featurize()`
        * non_iter_vars: list[str].
            * return(tuple) starts with [config_dict[v] for v in non_iter_vars] before iter_vars.
        
    method:
        * iter_config: return iterator
            `return product(*self.iterate_lists)`
    
    config_dic example:
        {
            "src": str,
            "unweighted": list["uw" | "w"],
            "feature": list[str],
            "stat": list[str]
        }
    """
    def __init__(self, config_dict: dict, non_iter_vars:list[str] = ["src"], iter_vars:list[str] = ["feature", "stat"]):
        self.config = config_dict
        self.iterate_lists = [[config_dict[v]] for v in non_iter_vars]
        self.iterate_lists = self.iterate_lists + [self.config[var] for var in iter_vars]
        
        self.shape = map(len, self.iterate_lists)
        self.len = math.prod(self.shape)

    def __len__(self):
        return self.len
    
    def iter_config(self) -> Iterator[tuple[str, ...]]:
        return product(*self.iterate_lists)
    
if __name__ == "__main__":
    testdict={
        "src": "src",
        "unweighted": "ABC",
        "feature": [1,2,3],
        "stat": "abc"
    }
    a = ConfigSingleSource(testdict)
    for i in a.iter_config():
        print(i)