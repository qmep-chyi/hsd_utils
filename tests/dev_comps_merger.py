#%%
import importlib.resources as resources
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from pymatgen.core.periodic_table import Element
from pymatgen.core import Composition
from scipy.spatial.distance import cdist

from hsdu.dataset import Dataset, D2TableDataset
from hsdu.utils.conversion_utils import norm_fracs, almost_equals_pymatgen_atomic_fraction
#%%
hsdset_path="some_path_to_dataset"
hsd = Dataset(hsdset_path)
hsd.df["comps_pymatgen"]
from hsdu.utils.conversion_utils import element_list_iupac_ordered
elements_ordered_iupac = element_list_iupac_ordered(hsd.elemental_set)

#%%
def encode_onehot_frac(comp:Composition, elements_ordered_iupac):
    elements = comp.elements
    fracs = norm_fracs(comp, elems=elements)
    
    el_symbols = [el.symbol for el in elements]

    onehot_frac = np.zeros(len(elements_ordered_iupac))

    for el, fr in zip(el_symbols, fracs):
        onehot_frac[elements_ordered_iupac.index(el)]=fr
    return onehot_frac

def decode_onehot_frac(onehot_frac:list, elements_ordered_iupac):
    
    comp_string=""
    for i, frac in enumerate(onehot_frac):
        if frac>0.0:
            el=elements_ordered_iupac[i]
            comp_string+=f"{el}{frac}"
    return Composition(comp_string)
#%%

comps_onehot_fracs=[]

for idx, row in hsd.df.iterrows():
    onehot_frac=encode_onehot_frac(row["comps_pymatgen"], elements_ordered_iupac)
    comps_onehot_fracs.append(onehot_frac)

hsd.df["comps_onehot_frac"]=comps_onehot_fracs

#%%
# test

for idx, row in hsd.df.iterrows():
    d=decode_onehot_frac(row["comps_onehot_frac"], elements_ordered_iupac)
    assert almost_equals_pymatgen_atomic_fraction(row["comps_pymatgen"], d, rtol=1e-9)

#%%
# Chebyshev
# chebyshev_dist = $L_{\infty}$=$\max_i |a_i - b_i|$
cdist([hsd.df.loc[0,"comps_onehot_frac"]],[hsd.df.loc[1,"comps_onehot_frac"]],"chebyshev")
#%%
# L1 dist
cdist([hsd.df.loc[0,"comps_onehot_frac"]],[hsd.df.loc[1,"comps_onehot_frac"]],"cityblock")