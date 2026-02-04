"""
utility functions to convert datatables.

Todo: refactor process_target(too long)
"""
from typing import Optional

import pandas as pd
import numpy as np
from hsdu.parsers import parse_value_with_uncertainty, InvalidTcException
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element, ElementBase


class OneHotFracCodec():
    def __init__(self, whole_elements_set: Composition | list[str] |set[str]):
        self.element_set_iupac: list[str]=element_list_iupac_ordered(whole_elements_set)
        self.element_index = {
            el: i for i, el in enumerate(self.element_set_iupac)
        }

    def encode(self, comp:Composition) -> list[float]:
        elements = comp.elements
        fracs = norm_fracs(comp, elems=elements)
        
        el_symbols = [el.symbol for el in elements]

        onehot_frac = [0.0]*len(self.element_set_iupac)

        for el, fr in zip(el_symbols, fracs):
            onehot_frac[self.element_index[el]]=fr
        return onehot_frac

    def decode(self, one_hot_frac:list[float]) -> Composition:
        comps_dict={}
        for i, frac in enumerate(one_hot_frac):
            if frac>0.0:
                el=self.element_set_iupac[i]
                comps_dict[el]=frac
        return Composition(comps_dict)
    
def element_list_iupac_ordered(elems: list[str] |list[ElementBase]|set[str]|set[ElementBase]) -> list[str]:
  if isinstance(elems, set):
    elems_out=list(elems)
  elif isinstance(elems, list):
    elems_out=elems
  else:
    raise ValueError(f"invalid type: {elems}")

  elems_out = [el.symbol if isinstance(el, ElementBase) else el for el in elems_out]
  elems_out = sorted(elems_out, key=lambda s: Element(s).iupac_ordering)
  return elems_out

def norm_fracs(comp: Composition | str, elems: Optional[list]=None, norm:bool=True):
    if isinstance(comp, str):
        comp = Composition(comp)
    fracs=[]
    if elems is not None:
        elems_to_iter=elems
    else:
        elems_to_iter=comp.elements
    for el in elems_to_iter:
        fracs.append(comp.get_atomic_fraction(el))
    return fracs/np.sum(fracs)

def almost_equals_pymatgen_atomic_fraction(a: Composition, b: Composition, rtol: float = 0.1)->bool:
    if set(a.elements)!=set(b.elements):
        return False
    a_fracs=norm_fracs(a, elems=a.elements)
    b_fracs=norm_fracs(b, elems=a.elements)
    return np.allclose(a_fracs, b_fracs, rtol=rtol)

def series_comps_gen(el_symbols:list[str], func, inps):
    """generate chemical composition string from..
    func(inps[idx]) should return fractions(float) list, same length with el_symbols.

    example usage:
        ```python
        inps=[0.88, 0.76, 0.65, 0.54, 0.43, 0.33, 0.21, 0.13, 0.04]
        series_comps_gen(el_symbols=['Ta', 'Nb', 'Zr', 'Hf', 'Ti'], func=(lambda x: [(1-x)/2]*2+[x/3]*3), inps=inps)
        >>> [0.06, 0.06, 0.29333333, 0.29333333, 0.29333333]\
            [0.12, 0.12, 0.25333333, 0.25333333, 0.25333333]\
            [0.175, 0.175, 0.21666667, 0.21666667, 0.21666667]\
            ...
        ```
        
    """
    for i in inps:
        fracs = [f"{fr:.08}" for fr in func(i)]
        fracs = f"[{', '.join(fracs)}]"
        print(fracs)
    return [func(i) for i in inps]

def process_targets(
        df: pd.DataFrame, 
        targets: list[str],
        non_sc_rule: str | None = "old", # "old" or "nan"
        exception_row: Optional[str] = "Exceptions",
        valid_targets: tuple[str, ...] = ("avg_Tc", "max_Tc", "std_Tc", "min_Tc"),
        return_num_tcs: bool=False,
        tc_cols=("Tc(K).resistivity.mid", "Tc(K).magnetization.mid", "Tc(K).resistivity.None", "Tc(K).magnetization.onset", "Tc(K).magnetization.None", "Tc(K).resistivity.zero", "Tc(K).specific_heat.mid", "Tc(K).other.None", "Tc(K).resistivity.onset", "Tc(K).specific_heat.onset", "Tc(K).specific_heat.None", "Tc(K).magnetization.zero", "Tc(K).other.onset", "Tc(K).other.mid", "Tc(K).specific_heat.zero"),
        process_dict: bool=False
        ) -> pd.DataFrame:
    """process targets, return a new pd.DataFrame
    
    arguments:
        * return_num_tcs: add "num_valid_tc" column.
    """
    #checkout targets
    assert all([t in valid_targets for t in targets]), NotImplementedError(f"current valid targets:{valid_targets}")
    target_array = []
    if return_num_tcs:
        targets.insert(0, "num_valid_tc")
    for row_idx, row in df.iterrows():
        if exception_row:
            assert pd.isna(row[exception_row])
        # gather tcs, not_passed_tc_cols.
        tcs=[]
        not_none_tc_cols=[]
        for key in tc_cols:
            val=row[key]
            if val is None or pd.isna(val):
                pass # most tc_cols are null.
            elif isinstance(val, (float, int)):
                tcs.append(parse_value_with_uncertainty(str(val)))
                not_none_tc_cols.append(key)

            elif isinstance(val, str):
                try:
                    parsed_tc=parse_value_with_uncertainty(val, process_dict=process_dict)
                except InvalidTcException as e:
                    e.add_note(f"key: {key} is not a string, not nan. row_idx: {row_idx}, comp:{row.get('comps_pymatgen', 'unknown comps')}, tcs:{tcs}, not_passed_tc_cols: {[row[ex_col] for ex_col in not_none_tc_cols]}, pd.isna: {pd.isna(val)}")
                    raise e
                if parsed_tc:
                    tcs.append(parsed_tc)
                    not_none_tc_cols.append(key)
                else:
                    assert parsed_tc is None
            else:
                raise TypeError(f"type(val): {type(val)} of val: {val} should be in (float, int, str)")
        # update target_array
        if non_sc_rule=="old":
            row_target_default_mean = 0.1 #0.1 is an arbitrary offset for non_sc
            row_target_default_std = 0.8 # 0.8 is an arbitrary standard deviation, make `offset+2sigma
        elif non_sc_rule=="nan":
            row_target_default_mean = np.nan
            row_target_default_std = np.nan
        else:
            raise NotImplementedError(f"non_sc_rule: {non_sc_rule}")
        row_target = []
        num_tc_row=len(tcs)
        if return_num_tcs:
            row_target.append(num_tc_row)
        if num_tc_row==0:
            print(f"no valid tc parsed. assign Tc=0 for row_idx: {row_idx}, comp:{row.get('comps_pymatgen', 'unknown comps')}, tcs:{tcs}, not_passed_tc_cols: {[row[ex_col] for ex_col in not_none_tc_cols]}")
            for target in targets:
                if target in ("avg_Tc", "max_Tc", "min_Tc"):
                    row_target.append(row_target_default_mean) 
                elif target=="std_Tc":
                    row_target.append(row_target_default_std) 
                elif target=="num_valid_tc":
                    pass
                else:
                    raise ValueError(f"target: {target} is not valid")
            assert len(row_target)==len(targets)
        elif num_tc_row==1:
            for target in targets:
                if target in ("avg_Tc", "max_Tc", "min_Tc"):
                    row_target.append(tcs[0]["mean"]) # it is mean of a single Tc value!!
                elif target=="std_Tc":
                    row_target.append(tcs[0]["std"])
                elif target=="num_valid_tc":
                    pass
                else:
                    raise ValueError
            assert len(row_target)==len(targets)
        else:
            for target in targets:
                if target == "avg_Tc":
                    row_target.append(np.mean([tc["mean"] for tc in tcs]))
                elif target == "max_Tc":
                    row_target.append(np.max([tc["mean"] for tc in tcs]))
                elif target == "std_Tc":
                    averaging_std = np.std([tc["mean"] for tc in tcs])
                    propagated_std = np.sum([tc["std"] for tc in tcs])/num_tc_row
                    row_target.append(averaging_std if averaging_std>propagated_std else propagated_std)
                elif target == "min_Tc":
                    row_target.append(np.min([tc["mean"] for tc in tcs]))
                elif target=="num_valid_tc":
                    pass
                else:
                    raise ValueError
            assert len(row_target)==len(targets)
        target_array.append(row_target)
    assert len(target_array)==len(df)
    return pd.DataFrame(data=target_array, columns=targets, dtype=float)