#%%
import importlib.resources as resources
from pathlib import Path
from typing import cast

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from pymatgen.core.periodic_table import Element, ElementBase
from pymatgen.core import Composition
from scipy.spatial.distance import cdist

from hsdu.dataset import Dataset, D2TableDataset
from hsdu.utils.conversion_utils import norm_fracs, almost_equals_pymatgen_atomic_fraction, element_list_iupac_ordered
#%%
hsdset_path="/home/chyi/hsd_utils/tests/temp_devs/hesc_dataset_16dec2025 - DataTable.csv"
hsd = Dataset(hsdset_path)

#%%
# test - encode / decode to comps_onehot_frac

for idx, row in hsd.df.iterrows():
    a= hsd.pymatgen_comps(idx)
    onehot_frac = hsd.onehot_frac.encode(a)
    d=hsd.onehot_frac.decode(onehot_frac)
    assert almost_equals_pymatgen_atomic_fraction(a, d, rtol=1e-9)

#%%
# Chebyshev
# chebyshev_dist = $L_{\infty}$=$\max_i |a_i - b_i|$ test
print(f'frac 0:{hsd.df.loc[[0,1],"elements"]}{hsd.df.loc[[0,1],"elements_fraction"]}')
print(f'pymatgen_comps(0):{hsd.pymatgen_comps(0)}')
print(f'pymatgen_comps(1):{hsd.pymatgen_comps(1)}')
onehot_frac0 = hsd.onehot_frac.encode(hsd.pymatgen_comps(0))
onehot_frac1 = hsd.onehot_frac.encode(hsd.pymatgen_comps(1))

#onehot_frac1 = cast(OneHotFrac, hsd.df.at[1, "comps_onehot_frac"]).to_numpy().reshape(1,-1)
print(cdist([onehot_frac0],[onehot_frac1],"chebyshev"))
#%%
# L1 dist
print(cdist([onehot_frac0],[onehot_frac1],"cityblock"))

#%%
