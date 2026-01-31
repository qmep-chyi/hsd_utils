"""some precedures for dataset.Dataset()

CellParser: xlsx cell Parsers.
    * ElemParser
    * FracParser

other Functions iterate DataFrame rows:
    * filter_row: filter rows with specific condition, return index to filter
    * 

Todo:
    - target_cols: hard coded
    - non_sc_rule: hard coded
    - vals = vals["nominal"]: hard coded
    - if val== 'Non-superconducting': hard coded

"""
from abc import ABC, abstractmethod
from typing import Callable
import re
import math
import ast
import fractions

from pymatgen.core.periodic_table import Element
import pandas as pd

class InvalidTcException(Exception):
    pass

class CellParser(ABC):
    """xlsx cell parser
    policy = | elements | number: how to parse the cell.
    
    when policy=object
    * Dictionary to dictionary, list to list
    * sometimes, 'Key' strings are not wrapped with quotation marks \
        (Should I handle this here?)
    * numbers in string format
        * just string
        * fractional number
        * (not implemented)with uncertainty
        * (not implemented) an inequations like '<2' 
    """
    def __init__(self, policy: str="object"):
        self.policy=policy

    @abstractmethod
    def parse(self, inp: str) -> any:
        raise NotImplementedError("self.parse()")
    
    def num_string_parser(self, v: str, uncertainty: bool = False, as_float: bool = True):
        """parsing string of a number

        args:
            * inp: string of a number. It can be a fraction or floats with uncertainty.
        """
        assert not uncertainty, NotImplementedError(uncertainty)
        assert as_float, NotImplementedError(as_float)
        
        if "(" in v: 
            # number with uncertainty. then, it cannot be a fraction
            assert v.strip()[-1]==")",v.strip()[-1]
            v_unc = v[v.find("("):-1] # uncertainty not implemented
            v_mean=ast.literal_eval(v[:v.find("(")])
        elif "/" in v:
            frac_pos=v.find("/")
            numerator = ast.literal_eval(v[:frac_pos])
            denominator = ast.literal_eval(v[frac_pos+1:])
            assert isinstance(numerator, int), numerator
            assert isinstance(denominator, int), denominator
            v_mean=fractions.Fraction(numerator, denominator)
        else:
            v_mean=float(v)

        return float(v_mean)
    
class ElemParser(CellParser):
    """
    parsing elements columns of the dataset
    """
    def __init__(self, policy="elem"):
        super().__init__(policy)

    def parse(self, inp: str) -> list[str]:
        elements_in_ptable = [el.symbol for el in Element]
        out:list[str] = ast.literal_eval(inp)
        assert all([x in elements_in_ptable for x in out]), f"Elements list {out} should include only elements symbols"
        return out

class FracParser(CellParser):
    """parse string of frac lists"""
    def __init__(self, policy="frac"):
        super().__init__(policy)

    def parse(self, inp, multiphase_rule: str = "nominal", as_float: bool = True):
        """ parse elemental_fractions as list of numbers
        
        args:
            * multi_fracs_rule:
                if "nominal", use fracs["nominal"].
        """
        inp = ast.literal_eval(inp)
        out: list[float] = []
        if isinstance(inp, dict):
            if multiphase_rule == "nominal":
                inp = inp["nominal"]
            else: 
                raise NotImplementedError(multiphase_rule)
        for v in inp:
            if isinstance(v, str):
                out.append(self.num_string_parser(v, as_float = as_float))
            else:
                assert isinstance(v, float) or isinstance(v, int), ValueError(v)
                out.append(float(v))
        assert isinstance(out, list)
        assert len(inp) == len(out)
        return out

# other functions that requires iterate whole data

def parse_value_with_uncertainty(s: str, 
                                 non_sc_strings=['Non-superconducting', 'non-SC'],
                                 process_dict:bool=False
                                 ):
    """
    Parse a number string with optional uncertainty in parentheses and return (mean, std).
    Examples:
      "4.563(8)"      -> (4.563, 0.008)
      "4.315"         -> (4.315, 1e-3 / sqrt(12))   # from rounding to last digit
      "-12.0"         -> (-12.0, 1e-1 / sqrt(12))
      "7"             -> (7.0, 1 / sqrt(12))        # rounded to integer
    """
    

    #handle exceptions
    assert not process_dict, NotImplementedError
    if '"' in s or "'" in s:
        if isinstance(ast.literal_eval(s), dict):
            return None
        else:
            s=s.replace("'","").replace('"',"")
    s_clean=s.strip()

    if s_clean[0]=="<":
        # non-SC observed on measure. currently no usage implemented
        return None
    elif s_clean in non_sc_strings:
        return None
    elif s_clean[0] in ["≈", "~"]:
        s_clean=s_clean[1:]

    # Extract a trailing "(digits)" if present
    m_unc = re.search(r"\((\d+)\)\s*$", s_clean)
    unc_digits = None
    if m_unc:
        unc_digits = m_unc.group(1)
        s_core = s_clean[:m_unc.start()].strip()
    else:
        s_core = s_clean

    # Split core into mantissa and optional exponent
    m = re.fullmatch(r"([+\-]?(?:\d+(?:\.\d*)?|\.\d+))", s_core)
    if not m:
        raise InvalidTcException(f"Could not parse numeric value from: {s!r}")

    mantissa_str = m.group(1)
    exponent = 0

    # Mean as normal float
    mean = float(mantissa_str) * (10 ** exponent)

    # Count decimals in the mantissa (digits after the dot)
    if "." in mantissa_str:
        decimals = len(mantissa_str.split(".")[1])
    else:
        decimals = 0

    # Compute std
    if unc_digits is not None:
        # Parentheses uncertainty applies to the last shown decimals of the mantissa
        # std = (integer in parens) * 10^(exponent - decimals)
        std = int(unc_digits) * (10 ** (exponent - decimals))
    else:
        # No uncertainty: assume rounding to last decimal place
        # ULP (one unit in last place) at this magnitude:
        ulp = 10 ** (exponent - decimals)
        # Standard deviation of uniform error over width=ULP
        std = ulp / math.sqrt(12.0)

    return {"mean":float(mean), "std": float(std)}

def mask_exception(df: pd.DataFrame, rule: Callable[[any], bool], filter_col: str = "Exceptions", print_count = True) -> pd.Series:
    mask = df.apply(rule[filter_col], axis=1)
    if print_count:
        print(f"mask.value_counts: {mask.value_counts()}")
    return mask
