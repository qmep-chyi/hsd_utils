"""load in-house dataset

"""

import json
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
import importlib.resources as resources
import string
from typing import Optional, cast

import pandas as pd
import numpy as np
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

from hsdu.parsers import CellParser, FracParser, ElemParser
from hsdu.utils.utils import config_parser
from hsdu.utils.conversion_utils import almost_equals_pymatgen_atomic_fraction, norm_fracs, OneHotFracEncoder, element_list_iupac_ordered


#from matminer.featurizers.composition import composite

class BaseDataset(ABC):
    """Base Class for Dataset classes, 
    
    argument:
        * drop_cols: drop these columns
        * exception_col: column to simply except some rows.
             e.g.) boolean column
    """
    def __init__(
            self, data_path: Optional[Path | str],
            comps_pymatgen_col: str="comps_pymatgen",
            drop_cols: Optional[list[str]] = None,
            exception_col: Optional[str | list[str]] = None,):
        if drop_cols is None:
            self.drop_cols = []
        else:
            self.drop_cols = drop_cols
        self.dset_path: Path = Path(".") if data_path is None else Path(data_path)
        self.df: pd.DataFrame = pd.DataFrame()
        self.exception_col = exception_col
        self.onehot_frac:OneHotFracEncoder
        self.elemental_set: list[str]

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

    def parse_col(self, col: str, cell_parser: CellParser, to_list: bool, data_type: type = str):
        parsed_rows = []
        for _, row in self.df.iterrows():
            parsed_rows.append(cell_parser.parse(row[col]))
        if to_list:
            raise NotImplementedError
            #self.dataframe[col].astype(data_type)
        return parsed_rows

    def pymatgen_duplicates(self, other_df:Optional[pd.DataFrame]=None, save_dir=None, exception_map:Optional[dict]=None, rtol=0.1, return_dict:bool=True):
        """
        compare elements and elements_fraction, find duplicates. 

        update attributes;
            * `self.duplicated_comps_group`(dictionary)
            * `self.duplicated_comps` (set)
        to compare, elements fractions are always normalized 
            - to meet `sum(fractions)==1.0`

        arguments:
            * other_df: Optional[pd.DataFrame].
                * if None, compare in-between (default)
                * if not None, other_df should have comps_pymatgen_col column().
                    * **note that it will overwrite `self attributes`**
        
        notes:
            * pymatgen.core.Composition().almost_equals() does not normalizes as I wanted, So I couldn't use that.

        arguments 
        other_df:Optional[pd.dataframe]=None
        """
        comps_pymatgen_col = self.comps_pymatgen_col
        self.duplicated_comps_group={}
        self.duplicated_comps=set()
        if other_df is None:
            df1=self.df
        else:
            #assert type(self).__name__=='XuTestHEA', "compare with other_df is implemented only for XuTestHEA"
            df1=other_df
        for idx0, row0 in self.df.iterrows():
            if idx0 not in self.duplicated_comps:
                duplicated_row={}
                idx1_start = 0 if other_df is not None else idx0 # when compare self, (i,j)==(j,i)
                row0_compare=cast(Composition, row0[comps_pymatgen_col])
                if exception_map is not None:
                    if idx0 in exception_map:
                        row0_compare=df1.loc[exception_map[idx0], comps_pymatgen_col]

                for idx1, row1 in df1[idx1_start:].iterrows():
                    if almost_equals_pymatgen_atomic_fraction(row1[comps_pymatgen_col], row0_compare, rtol=rtol):
                        duplicated_row[idx1]=row1[comps_pymatgen_col]

                for idx1, row1 in df1[idx1_start:].iterrows():
                    if almost_equals_pymatgen_atomic_fraction(row1[comps_pymatgen_col], row0[comps_pymatgen_col], rtol=rtol):
                        duplicated_row[idx1]=row1[comps_pymatgen_col]
                minimum_length = 2 if other_df is None else 1 #when self compare, it includes self.
                if len(duplicated_row)>=minimum_length:
                    print(f"idx0, duplicates: {idx0}, {duplicated_row}")
                    self.duplicated_comps_group[idx0]={
                        k: str(v) for k, v in duplicated_row.items()
                        }
                    if other_df is None:
                        # when other_df, `idx0 not in self.duplicated_comps` should be false always
                        self.duplicated_comps.update(list(duplicated_row.keys()))
                    else:
                        # 
                        self.duplicated_comps_group[idx0][f"self_{idx0}"]=str(row0[comps_pymatgen_col])
                        if exception_map is not None:
                            if idx0 in exception_map:
                                self.duplicated_comps_group[idx0][f"replace_to_nominal_comp_idx0918({exception_map[idx0]})"]=str(row0_compare)

        if save_dir is not None:
            with open(save_dir, 'w', encoding="utf-8") as f:
                json.dump(self.duplicated_comps_group, f, indent=4, ensure_ascii=False)
    
    def validate_by_composition(self, rtol:float=0.001, 
                                composition_col="composition"):
        """compare `composition` column with `comps_pymatgen` column

        try to parse self.df[composition_col] by
        Composition() from `pymatgen.core.composition`,
        compare with self.df[self.comps_pymatgen_col].

        Note that it is supplemental validation,
        while only simple cases are parsed by Composition()

        args:
            - composition_col:str="composition"
                - self.df[composition_col]:
                    the composition string of the source (literature) as is,
                    may have complex format (e.g. variables like x)
        """
        allowed = set(string.ascii_letters + string.digits + '.')
        for idx, row in self.df.iterrows():
            comp=None
            if set(row["composition"]) <= allowed:
                try:
                    comp=Composition(row["composition"])
                except ValueError as e:
                    warnings.warn(f'skip validate by Composition() from \
                                  `pymatgen.core.composition` of entry: \
                                  comp: {row["composition"]} on idx: {idx}\
                                    raised ValueError:\n {e}')
                    comp=None
            if comp is not None:
                #hard-coded exceptions. should be refactored.
                if idx in list(range(27, 40)):
                    exception_case="Oxygen not included on the composition (hard coded exceptions)"
                    warnings.warn(f"skip {idx}, {exception_case}")
                    pass
                elif idx in list(range(88, 93)):
                    exception_case="nominal - actual diff (hard coded exceptions)"
                    warnings.warn(f"skip {idx}, {exception_case}")
                    pass
                else: #start validation
                    if almost_equals_pymatgen_atomic_fraction(row["comps_pymatgen"], comp, rtol=rtol):
                        pass
                    else:
                        assert row["comps_pymatgen"].elements==comp.elements
                        warnings.warn(f"idx: {idx}, comp:{comp}, comps_pymatgen:{row['comps_pymatgen']}")
                        print(f"while norm_fracs(comp):{norm_fracs(comp)}, norm_fracs_comps_pymatgen:{norm_fracs(row['comps_pymatgen'])}")

                    
    def add_duplicated_comps_column(self, criteria_rule: str, inplace=True):
        assert criteria_rule in ['single_ref', 'dataset'], NotImplementedError(criteria_rule)
        duplicate_groups=[] # will be a new column, group name = first instance idx(in self.dataset.duplicated_comps_group.keys)
        for idx0, row0 in self.df.iterrows():
            group_row=np.nan
            if idx0 in self.duplicated_comps_group.keys():
                for idx1 in self.duplicated_comps_group[idx0].keys():            
                    cite0=row0["full citation"]
                    cite1=self.df.loc[idx1, "full citation"]
                    group_row = idx0 if cite0==cite1 else np.nan
            duplicate_groups.append(group_row)
        if inplace:
            self.df['duplicated_group']=duplicate_groups
        return duplicate_groups
    
    def assign_dtypes(self, dtype:type=str):
        for col in self.df.columns:
            if self.df.col.dtype not in (list, dict, float, str):
                self.df[col] = self.df[col].astype(dtype)

class D2TableDataset(BaseDataset):
    """Base Class for Dataset classes, load xlsx file
    
    load csv, generate features and an export array for ML.

    Parameters:
        * xls_path: path to the the xlsx file
        * notebook: Excel sheet name to load dataset

    See Also(inherited by):
        - Dataset()
        - XuDataset(): Load Xu 2025 HEAs used to validate.
    """
    def __init__(
            self, dset_path: Path | str,
            notebook: str | None = None,
            drop_cols: Optional[list[str]] = None,
            index_col: Optional[str] = None,
            exception_col: Optional[str | list[str]] = "Exceptions",
            encode_onehot_fracs:bool=True):
        if drop_cols is None:
            self.drop_cols = []
        else:
            self.drop_cols = drop_cols

        # pd.read_csv or pd.read_xls creates a new index column if not specified
        if index_col is None:
            self.index_col=None
        else:
            self.index_col=index_col

        if isinstance(dset_path, str):
            dset_path = Path(dset_path)
        if dset_path.is_absolute():
            pass
        else:
            if dset_path.is_file():
                pass
            else:
                with resources.as_file(resources.files("hsdu.data").joinpath(str(dset_path))) as pth:
                    dset_path = pth
                assert dset_path.is_file(),FileNotFoundError

        super().__init__(dset_path, drop_cols = drop_cols)
        self.dset_path: Path = self.dset_path
        self.sheet = notebook
        self.maxlen: Optional[int] = None

        df:pd.DataFrame = self.load_data()
        if exception_col is not None:
            df = df[df[exception_col].apply(pd.isna)]
        else:
            pass
        self.df: pd.DataFrame = df.reset_index(drop=True)

        if encode_onehot_fracs:
            self.elemental_stats()

    def elemental_stats(self):
        """set of elements in the dataset"""
        self.parsed_fracs_rows = self.parse_frac_col()
        self.parsed_elements_rows = self.parse_elements_col()
        elems=set()
        for parsed_elems_row in self.parsed_elements_rows:
            elems.update(set(parsed_elems_row))
        self.elemental_set=element_list_iupac_ordered(elems)

        self.onehot_frac=OneHotFracEncoder(self.elemental_set)

        comps_dict_rows = self.onehot_frac_dicts()
        onehot_frac_rows = [self.onehot_frac.encode(Composition(dic)) for dic in comps_dict_rows]

        onehot_df = pd.DataFrame(onehot_frac_rows, index=self.df.index, columns=self.elemental_set)
        self.df=pd.concat([self.df, onehot_df], axis=1)
    def onehot_frac_dicts(self) -> list[dict[str, float]]:
        comps_dict_rows=[]
        for els, fracs in zip(self.parsed_elements_rows, self.parsed_fracs_rows):
            comps_dict_rows.append(self.onehot_frac_dict(els, fracs))
        return comps_dict_rows
    
    def onehot_frac_dict(self, els, fracs) -> dict[str, float]:
        return {el:frac for el, frac in zip(els, fracs)}
    
    def pymatgen_comps(self, idx) -> Composition:
        comps_pymatgen = Composition(zip(self.elemental_set, self.df.loc[idx, self.elemental_set]))
        return comps_pymatgen

    def load_data(self) -> pd.DataFrame:
        if self.dset_path.suffix==".csv":
            df = pd.read_csv(self.dset_path,
                            nrows=self.maxlen,
                            index_col=self.index_col)
        else:
            raise ValueError("read only csv file(should have suffix).")

        drop_cols_remaining=[]
        if len(self.drop_cols)>0:
            for col in self.drop_cols:
                if col in df.columns:
                    drop_cols_remaining.append(col)
        return df.drop(columns=drop_cols_remaining)

    def parse_elements_col(self, colname: str="elements")->list[list[str]]:
        """parse string of a csv cell with elements in list form"""
        cell_parser = ElemParser()
        return self.parse_col(colname, cell_parser, False)

    def parse_frac_col(self, colname: str="elements_fraction")->list[list[float]]:
        """parse string of a csv cell with fractions in list form"""
        cell_parser = FracParser()
        return self.parse_col(colname, cell_parser, False)
    
class Dataset(D2TableDataset):
    """
    attributes:
        * exception_col: simple filter by specific column values

    methods:
        * validate_elem_frac_length:
    
    """
    def __init__(self, 
                 xls_path, 
                 config: str | dict | Path = "default",
                 drop_cols: Optional[list[str]] = None,
                 exception_col: Optional[str | list[str]] = "Exceptions"):
        self.config = config_parser(config, mode="dataset")
        super().__init__(
            xls_path, 
            notebook = self.config.get("sheetname"), 
            drop_cols = self.config.get("drop_cols", drop_cols), 
            index_col= self.config.get("index_column"),
            exception_col=self.config.get("exception_col", exception_col))
        #self.validate_elem_frac_length()

    def validate_elem_frac_length(self):
        raise DeprecationWarning
        assert self.df.apply(
            lambda x: (len(x["elements"])==len(x["elements_fraction"])), axis=1).all(),\
            self.df.loc[self.df.apply(
            lambda x: len(x["elements"])!=len(x["elements_fraction"]), axis=1)]