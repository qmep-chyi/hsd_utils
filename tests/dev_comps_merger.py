#%%
import numpy as np
import pandas as pd
from typing import Literal

from scipy.spatial.distance import cdist
from pymatgen.core import Composition

from hsdu.dataset import Dataset, D2TableDataset
from hsdu.utils.conversion_utils import almost_equals_pymatgen_atomic_fraction, element_list_iupac_ordered
from hsdu.utils.duplicate import make_duplicates_group, distance_matrix, compare_dupl_groups, compare_dupl_groups_old2new


#%%
hsdset_path=r"C:\Users\hms_l\hsd_utils\temp_devs\hsd16dec_from205to210.csv"
hsd = Dataset(hsdset_path)
hsd.pymatgen_duplicates(rtol=0.02)
