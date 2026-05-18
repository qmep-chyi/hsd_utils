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
    def __init__(self):
        warnings.warn(r"This data is not covered by the license of this repository but by Xu et al., Predicting superconducting temperatures with new hierarchical neural network AI model. Front. Phys. 20, 14205 (2025) DOI:10.15302/frontphys.2025.014205", ExternalDataWarning)
        with resources.files("hsdu.data.miscs") /"xu2025_test_HEAs.csv" as path:
            super().__init__(dset_path=path, exception_col=None,
                            parse_pymatgen_comps_col='formula')
        
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

        with resources.files("hsdu.data.miscs") /"preprocessed_supercon.csv" as path:
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