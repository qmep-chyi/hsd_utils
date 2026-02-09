#%%
import numpy as np
import pandas as pd
from typing import Literal
from difflib import get_close_matches

from scipy.spatial.distance import cdist
from pymatgen.core import Composition

from hsdu.dataset import Dataset, D2TableDataset
from hsdu.utils.conversion_utils import almost_equals_pymatgen_atomic_fraction, element_list_iupac_ordered
from hsdu.utils.duplicate import make_duplicates_group, distance_matrix, compare_dupl_groups, compare_dupl_groups_old2new


                    
hsdset_path=r"asdf"
hsd = Dataset(hsdset_path)

from difflib import SequenceMatcher
import re
refs = pd.read_json("asdfasdf")
#%% updating to 09feb2026 version; adding doi columns
ref_titles_alphanumeric_only = [re.sub(r'[^a-zA-Z0-9]', '', ref) for ref in refs['title'].to_list()]
short_cite0=""
full_cite0=""
short_cite1=""
full_cite1=""
for i, row in enumerate(hsd):
    short_cite1=row['short_cite']
    full_cite1=row['full citation']
    if short_cite1!=short_cite0:
        try:
            closest = get_close_matches(row['full citation'], refs['title'].to_list(), cutoff=0.6)[0]
        except IndexError as e:
            full_cite_alphanumeric_only = re.sub(r'[^a-zA-Z0-9]', '', row['full citation'])
            closest = get_close_matches(full_cite_alphanumeric_only, ref_titles_alphanumeric_only, cutoff=0.2)
            for j, closer in enumerate(closest):
                closest_index = ref_titles_alphanumeric_only.index(closer)
                print(f"j-th close: {refs['title'][closest_index]}")
                print(f"while first authors (from refs): {refs['author'][closest_index][0]}\n")
                #print(row)
            print(f"full citation: {row['full citation']}\n short_cite:{row['short_cite']}\n")
            
        short_cite0=short_cite1
        full_cite0=full_cite1
    elif short_cite1==short_cite0:
        assert full_cite0==full_cite1