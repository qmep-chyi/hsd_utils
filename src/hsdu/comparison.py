"""comparisons with literatures

XuDataset: Xu et al. (2025)

Todo:
    * duplicates; in __init__, `elem_list=[]..`

References
    - Xu, S. Predicting superconducting temperatures with new hierarchical neural network AI model. Front. Phys. 20, 14205 (2025).
"""
import importlib.resources as resources
import warnings
from typing import Optional

import pandas as pd
from pymatgen.core.composition import Composition

from hsdu.dataset import D2TableDataset, Dataset

class ExternalDataWarning(UserWarning):
    pass

class XuTestHEA(D2TableDataset):
    """Reproduced test HEA set of Xu et al. 2025.
    
    see Supplementary Table S1
    """
    def __init__(self, csv_path=None):
        warnings.warn(r"This data is not covered by the license of this repository but by Xu et al., Predicting superconducting temperatures with new hierarchical neural network AI model. Front. Phys. 20, 14205 (2025) DOI:10.15302/frontphys.2025.014205", ExternalDataWarning)
        if csv_path is None:
            with resources.as_file(resources.files("hsdu.data.miscs") /"xu2025_test_HEAs.csv") as path:
                csv_path = path

        super().__init__(dset_path=csv_path, exception_col=None,
                        parse_pymatgen_comps_col='formula', index_col="xu_index")
            
class StanevSuperCon(D2TableDataset):
    """load preprocessed SuperCon csv 
    
    * Preprocessed SuperCon table from Stanev et al. (2018)

    preprocess:
        * see `src.hsdudata.miscs.preprocess_supercon`
        * note that treshold_max_tc is hard coded, because it is close to the 5213 entries of xu et al
    """
    def __init__(self, drop_cols = None, exception_col = None,
                 maxlen: Optional[int] = None,
                 treshold_max_tc=12,
                 encode_onehot_fracs:bool=True):
        warnings.warn(r"This data is not covered by the license of this repository but Stanev, V. et al. Machine learning modeling of superconducting critical temperature. npj Comput Mater 4, 29 (2018). (doi: 10.1038/s41524-018-0085-8)", ExternalDataWarning)
        if not encode_onehot_fracs:
            raise NotImplementedError

        with resources.as_file(resources.files("hsdu.data.miscs") /"preprocessed_supercon.csv") as path:
            super().__init__(dset_path=path, drop_cols=drop_cols,
                            exception_col=exception_col,
                            index_col='index',
                            encode_onehot_fracs=encode_onehot_fracs, 
                            parse_pymatgen_comps_col='name')
        
        elem_list=[]
        frac_list=[]
        drop_rows=[]

        for idx, row in self._df.iterrows():
            # column Tc is renamed to Tc0, because of `Technetium (Tc)`
            if row["Tc0"]>treshold_max_tc:
                drop_rows.append(idx)
            else:
                try:
                    comp=Composition(row["name"])
                    row_fracs =[comp.get_atomic_fraction(comp.as_data_dict()["elements"][i])
                        for i in list(range(comp.as_data_dict()["nelements"]))]
                    elem_list.append(comp.as_data_dict()["elements"])
                    frac_list.append(row_fracs)
                except ValueError:
                    drop_rows.append(idx)
        self._df = self._df.drop(index=drop_rows, axis=0)
        self._df: pd.DataFrame = self._df.reset_index(drop=True)

        self.config=dict(targets=['Tc0'])
        if self.config['targets']==['Tc0']:
            assert 'Tc' in self._df.columns

if __name__=="__main__":
    # -----search xu2025 composition formula on merged dataset, check duplicated--------
    #   * does not consider which references it came from
    #   
    xu_test = XuTestHEA()
    dataset = Dataset(csv_path=r"C:\Users\chyi\draftsh2025\temp_devs\merged_dataset_forward.csv", config="default_forward.json", exception_col=None)

    #exception_map, as we currently use "nominal composition" only.
    exception_map={
        7:324,  # (nominal - dendrite_phase) xu(7)->319 in `index_0810`. `sample C5`
        20:326, # (nominal - dendrite_phase) xu(20)->321 in `index_0810``
    }
    xu_test.pymatgen_duplicates(dataset.df, rtol=0.1, save_dir="temp_devs/xu_to_0918_duplicates_rtol0_1.json", exception_map=exception_map)
    for i in range(len(xu_test.df)):
        print(list(xu_test.duplicated_comps_group[i].keys())[:-1])
    # print index_0810    
    for i in range(len(xu_test.df)):
        index_0810s=[]
        for key in xu_test.duplicated_comps_group[i].keys():
            if isinstance(key, int):
                if pd.isna(dataset.df.loc[key, 'index_0810']):
                    index_0810s.append("new_0918")
                else:
                    index_0810s.append(int(dataset.df.loc[key, 'index_0810']))
        print(index_0810s)

    # revert dictionary
    idx0918_to_xu_test={}
    for k, v in xu_test.duplicated_comps_group.items():
        for k1, _ in v.items():
            if isinstance(k1, int):
                idx0918_to_xu_test[k1]=k
            
    for i in range(len(dataset.df)):
        print(idx0918_to_xu_test.get(i, "false"))
    a=[1234,534] #placeholder..
