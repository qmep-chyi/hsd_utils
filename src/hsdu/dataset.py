"""load in-house dataset

"""

import json
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
import importlib.resources as resources
import string
from typing import Optional, cast
from collections.abc import Sequence

import pandas as pd
import numpy as np
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

from hsdu.parsers import CellParser, FracParser, ElemParser
from hsdu.utils.utils import config_parser
from hsdu.utils.conversion_utils import almost_equals_pymatgen_atomic_fraction, norm_fracs, OneHotFracCodec, element_list_iupac_ordered

class ElementsSymbolColWarning(UserWarning):
    """column name is a symbol of element"""
    pass

class InitRequriedError(Exception):
    """Some initialization not finished"""
    pass

#from matminer.featurizers.composition import composite

class BaseDataset(ABC, Sequence):
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
        self._df: pd.DataFrame = pd.DataFrame()
        self.exception_col = exception_col
        self.idx2aux: dict[dict[int, any]] = dict()#dictionaries sidecar for auxiliary informations that shares self._df.index
        self.column_sets:dict[dict[list[str]]] = dict( # set of column names in `self._df.columns``
            old_index=[],
            raw=[],
            processed=[],
            T_c=[],
            categorical_data=[],
            categorical_misc=[],
            annotations=[]
        )
        self.index: list[int] = None

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

    def check_index(self):
        """validate index of `self._df` and `values(self.idx2aux)`"""
        df_index=self._df.index.tolist()
        for k, v in self.idx2aux.items():
            assert set(list(v.keys()))==set(df_index), f"index of {k} != self._df.index"
        return True
    
    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        
        if i < 0:
            i+=len(self)

        if i < 0 or i >= len(self):
            raise IndexError(i)

        try:
            out = {k:v[i] for k, v in self.idx2aux.items()}
        except KeyError as e:
            raise IndexError(i) from e
        
        assert len(set(out.keys()) & set(self._df.loc[i, :].keys()))==0

        try:
            return out | dict(self._df.loc[i, :])
        except KeyError as e:
            raise IndexError(i) from e
    
    def __len__(self):
        return len(self._df)

    def parse_col(self, col: str, cell_parser: CellParser, to_list: bool, data_type: type = str):
        parsed_rows = []
        for _, row in self._df.iterrows():
            parsed_rows.append(cell_parser.parse(row[col]))
        if to_list:
            raise NotImplementedError
            #self.dataframe[col].astype(data_type)
        return parsed_rows
    
    def validation_index(self):
        """validation_index: index, df orders
        
        self.idx2aux["aux_dict_name"].keys() and self._df.index.tollist()
        is not intended to be changed.
        """
        # self._df.index is a rangeindex from 0.
        assert self._df.index.tolist()==list(range(len(self._df)))
        # all index matches with self.idx2aux
        assert self.check_index()
        # assert self.index, keys() not changed
        if self.index is None:
            self.index =  self._df.index.tolist()
        else:
            assert self.index==self._df.index.tolist()
        assert self.index==list(range(len(self.index)))
        assert all(list(v.keys())==self._df.index.tolist() for _, v in self.idx2aux.items())
        return True
    
    #def group_close_comps(self, dinfty_cutoff=0.01, d1_cutoff=0.02):

    def pymatgen_duplicates(self, other_df:Optional[pd.DataFrame | pd.Series | list]=None, save_dir=None, exception_map:Optional[dict]=None, rtol=0.1, return_dict:bool=True):
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
        warnings.warn("use make_duplicates_group from hsdu.utils.duplicate", DeprecationWarning)

        self.duplicated_comps_group={}
        self.duplicated_comps=set()

        assert other_df is None or isinstance(other_df, pd.DataFrame), NotADirectoryError

        if other_df is None:
            df1=self._df
            other_comps = self.idx2aux['comps_pymatgen'].copy()
        else:
            #assert type(self).__name__=='XuTestHEA', "compare with other_df is implemented only for XuTestHEA"
            df1=other_df
        for idx0, row0 in self._df.iterrows():
            if idx0 not in self.duplicated_comps:
                duplicated_row={}
                idx1_start = 0 if other_df is not None else idx0 # when compare self, (i,j)==(j,i)
                row0_compare=cast(Composition, self.idx2aux['comps_pymatgen'][idx0])

                for idx1, row1 in df1[idx1_start:].iterrows():
                    if almost_equals_pymatgen_atomic_fraction(other_comps[idx1], row0_compare, rtol=rtol):
                        duplicated_row[idx1]=other_comps[idx1]

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
                        self.duplicated_comps_group[idx0][f"self_{idx0}"]=str(self.idx2aux['comps_pymatgen'][idx0])
                        if exception_map is not None:
                            if idx0 in exception_map:
                                self.duplicated_comps_group[idx0][f"replace_to_nominal_comp_idx0918({exception_map[idx0]})"]=str(row0_compare)

        if save_dir is not None:
            with open(save_dir, 'w', encoding="utf-8") as f:
                json.dump(self.duplicated_comps_group, f, indent=4, ensure_ascii=False)
    
    def validate_by_composition(self, rtol:float=0.001, 
                                composition_col="composition"):
        """compare `composition` column with `comps_pymatgen` column

        try to parse self._df[composition_col] by
        Composition() from `pymatgen.core.composition`,
        compare with self._df[self.comps_pymatgen_col].

        Note that it is supplemental validation,
        while only simple cases are parsed by Composition()

        args:
            - composition_col:str="composition"
                - self._df[composition_col]:
                    the composition string of the source (literature) as is,
                    may have complex format (e.g. variables like x)
        """
        allowed = set(string.ascii_letters + string.digits + '.')
        for idx, row in self._df.iterrows():
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
                    if almost_equals_pymatgen_atomic_fraction(self.idx2aux['comps_pymatgen'][idx], comp, rtol=rtol):
                        pass
                    else:
                        assert self.idx2aux['comps_pymatgen'][idx].elements==comp.elements
                        warnings.warn(f"idx: {idx}, comp:{comp}, comps_pymatgen:{self.idx2aux['comps_pymatgen'][idx]}")
                        print(f"while norm_fracs(comp):{norm_fracs(comp)}, norm_fracs_comps_pymatgen:{norm_fracs(self.idx2aux['comps_pymatgen'][idx])}")

    def add_duplicated_comps_column(self, criteria_rule: str, inplace=True):
        """
        run self.pymatgen_duplicates() first!
        """
        warnings.warn("use make_duplicates_group from hsdu.utils.duplicate", DeprecationWarning)
        try:
            self.duplicated_comps_group
        except AttributeError as e:
            if "duplicated_comps_group" in str(e):
                raise InitRequriedError("run self.pymatgen_duplicates() first!")

        assert criteria_rule in ['single_ref', 'dataset'], NotImplementedError(criteria_rule)
        duplicate_groups=[] # will be a new column, group name = first instance idx(in self.dataset.duplicated_comps_group.keys)
        for idx0, row0 in self._df.iterrows():
            group_row=np.nan
            if idx0 in self.duplicated_comps_group.keys():
                for idx1 in self.duplicated_comps_group[idx0].keys():            
                    cite0=row0["full citation"]
                    cite1=self._df.loc[idx1, "full citation"]
                    group_row = idx0 if cite0==cite1 else np.nan
            duplicate_groups.append(group_row)
        if inplace:
            self._df['duplicated_group']=duplicate_groups
        return duplicate_groups
    
    def assign_dtypes(self, dtype:type=str):
        for col in self._df.columns:
            if self._df.col.dtype not in (list, dict, float, str):
                self._df[col] = self._df[col].astype(dtype)

class D2TableDataset(BaseDataset):
    """Base Class for Dataset classes, load csv file
    
    load csv, generate features and an export array for ML.

    arguments:
        * dset_path: path to the the dataset (csv) file
        * parse_pymatgen_comps_col:
            - if not None, just use 
                `pymatgen.core.Composition(self._df[parse_pymatgen_comps][idx])`
                as composition, instead of parsing 'elements' or 'elements_fraction'.

    See Also(inherited by):
        - Dataset()
        - XuDataset(): Load Xu 2025 HEAs used to validate.
    """
    def __init__(
            self, dset_path: Path | str,
            drop_cols: Optional[list[str]] = None,
            index_col: Optional[str] = None,
            exception_col: Optional[str | list[str]] = "Exceptions",
            encode_onehot_fracs:bool=True,
            parse_pymatgen_comps_col:str|None=None):
        
        self._df:pd.DataFrame
        self.onehot_codec: OneHotFracCodec # encode compositions to the one-hot style multiple columns.
        self.elemental_set: list[str] # `iupac ordered`` list of elements symbol included in the dataset

        if drop_cols is None:
            self.drop_cols = []
        else:
            self.drop_cols = drop_cols

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
        self.maxlen: Optional[int] = None

        df:pd.DataFrame = self.load_data()
        if exception_col is not None:
            df = df[df[exception_col].apply(pd.isna)]
        else:
            pass
        self._df: pd.DataFrame = df.reset_index(drop=True)

        if encode_onehot_fracs:
            elements_symbols = [i.symbol for i in Element]
            for key in self._df.columns:
                if key in elements_symbols:
                    warnings.warn(f"{key} in self._df.columns, renamed to {key+'0'}",
                                  category=ElementsSymbolColWarning)
                    self._df.rename(columns={key:str(key+'0')}, inplace=True)
            self.encode_onehot_fracs(parse_pymatgen_comps_col=parse_pymatgen_comps_col)

    def encode_onehot_fracs(self, inplace=True, parse_pymatgen_comps_col:str|None=None):
        """Initiallize elemental_set, encode composition to the onehot-like multiple columns
        
        arguments:
            - inplace: 
                - add 'elements_set' column. e.g. 'Sc-Hf-Ti-Ta-Nb' 
                - add len(self.elemental_set) columns for elements, encode fraction to self._df
                - overwrite attributes (self.-); 
                    onehot_codec, elemental_set
                - overwrite self.idx2aux[parsed_fracs, parsed_elements, comps_pymatgen]
                - update self.index
                - If false, raise NotImplementedError
            - composition_col:
                - if not None, pymatgen.core.Composition(self._df[composition.col][idx])\
                    instead of parsing 'elements' and 'elements_fracion'
        """
        assert inplace, NotImplementedError
        df_index =  self._df.index.copy().tolist()
        self.index = df_index if self.index is None else self.index

        if parse_pymatgen_comps_col is None:
            self.idx2aux["parsed_fracs"] = dict(zip(df_index, self.parse_frac_col()))
            self.idx2aux["parsed_elements"] = dict(zip(df_index, self.parse_elements_col()))
            pymatgen_comps_rows = [Composition(dic) for dic in self.onehot_frac_dicts()]
        else:
            pymatgen_comps_rows = [Composition(comp) for comp in self._df.loc[:, parse_pymatgen_comps_col]]
            self.idx2aux['parsed_fracs'] = dict()
            self.idx2aux['parsed_elements'] = dict()
            for idx, pymatgen_comps in enumerate(pymatgen_comps_rows):
                self.idx2aux['parsed_elements'][idx] = element_list_iupac_ordered(pymatgen_comps.elements)
                self.idx2aux['parsed_fracs'][idx] = [pymatgen_comps[el] for el in self.idx2aux['parsed_elements'][idx]]

        self.idx2aux["comps_pymatgen"] = dict(zip(df_index, pymatgen_comps_rows))
        self.elemental_set=self.compile_elements_set(self.idx2aux["parsed_elements"])
        self.onehot_codec=OneHotFracCodec(self.elemental_set)


        onehot_frac_rows = [self.onehot_codec.encode(Composition(dic)) for dic in pymatgen_comps_rows]

        onehot_df = pd.DataFrame(onehot_frac_rows, index=self._df.index, columns=self.elemental_set)
        onehot_df["elements_set"] = ["-".join(element_list_iupac_ordered(i.elements)) for i in pymatgen_comps_rows]
        self._df=pd.concat([self._df, onehot_df], axis=1)
        self.column_sets["onehot_elements"]=self.elemental_set
        assert self.validation_onehot_frac()
        assert self.validation_index()

    def validation_onehot_frac(self):
        """validation after self.encode_onehot_fracs.
        
        index, columns, df, idx2aux..."""
        # all encoded elements exists in iupac order
        first_elem_col=self._df.columns.tolist().index(self.elemental_set[0])
        last_elem_col=self._df.columns.tolist().index(self.elemental_set[-1])
        assert all(self._df.columns[first_elem_col:last_elem_col+1]==self.elemental_set)
        return True

    def compile_elements_set(self, parsed_elems_rows:list[list[str]]):        
        elems=set()
        for i in range(len(parsed_elems_rows)):
            elems.update(set(parsed_elems_rows[i]))
        return element_list_iupac_ordered(elems)

    def onehot_frac_dicts(self) -> list[dict[str, float]]:
        comps_dict_rows=[]
        for els, fracs in zip(self.idx2aux["parsed_elements"].values(), self.idx2aux["parsed_fracs"].values()):
            comps_dict_rows.append(self.onehot_frac_dict(els, fracs))
        return comps_dict_rows
    
    def onehot_frac_dict(self, els, fracs) -> dict[str, float]:
        return {el:frac for el, frac in zip(els, fracs)}
    
    def onehot_fracs(self) -> list[list[float]]:
        """return onehot_fracs of whoel dataset"""
        return self._df[self.column_sets["onehot_elements"]].to_numpy().tolist()

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
                 csv_path, 
                 config: str | dict | Path = "default",
                 drop_cols: Optional[list[str]] = None,
                 exception_col: Optional[str | list[str]] = "Exceptions"):
        self.config = config_parser(config, mode="dataset")
        super().__init__(
            csv_path, 
            drop_cols = self.config.get("drop_cols", drop_cols), 
            index_col= self.config.get("index_column"),
            exception_col=self.config.get("exception_col", exception_col))
        #self.validate_elem_frac_length()

    def validate_elem_frac_length(self):
        raise DeprecationWarning
        assert self._df.apply(
            lambda x: (len(x["elements"])==len(x["elements_fraction"])), axis=1).all(),\
            self._df.loc[self._df.apply(
            lambda x: len(x["elements"])!=len(x["elements_fraction"]), axis=1)]