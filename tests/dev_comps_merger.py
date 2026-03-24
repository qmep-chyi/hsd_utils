######## TODO: refactor required ########
#%%
#%%
# debudding xgboost_9mar.ipynb
from hsdu.dataset import Dataset, D2TableDataset
dataset_path = "/home/chyi/hsd_utils/tests/temp_devs/test_featurized_table.csv"
ftd255=D2TableDataset(dataset_path, exception_col=None, encode_onehot_fracs=False)
train202=D2TableDataset('/home/chyi/hsd_utils/tests/temp_devs/HESC_train_col450.csv', exception_col=None, encode_onehot_fracs=False, drop_cols=['comps_pymatgen'])
test49=D2TableDataset('/home/chyi/hsd_utils/tests/temp_devs/HESC_test_col450.csv', exception_col=None, encode_onehot_fracs=False, drop_cols=['comps_pymatgen'])
hsd = Dataset('/home/chyi/hsd_utils/src/hsdu/data/tests/full_dataset.csv', exception_col='Exceptions')
test49.encode_onehot_fracs(fixed_elements_set=hsd.elemental_set, rule_elements_set='overwrite')
dupl_group, idx2group=hsd.group_duplicates(other=test49, cityblock=0.01, msre=0.02, update_attrs=False, )
#%%

if False:
    import numpy as np
    import pandas as pd
    from typing import Literal
    from difflib import get_close_matches
    import warnings
    import json
    from pathlib import Path

    from scipy.spatial.distance import cdist
    from pymatgen.core import Composition

    from hsdu.dataset import Dataset, D2TableDataset
    from hsdu.utils.conversion_utils import almost_equals_pymatgen_atomic_fraction, element_list_iupac_ordered
    from hsdu.utils.duplicate import make_duplicates_group, distance_matrix, compare_dupl_groups, compare_dupl_groups_old2new


                        
    hsdset_path=r"/home/chyi/hsd_utils/tests/temp_devs/merged_dataset_forward - DataTable.csv"
    hsd = Dataset(hsdset_path)

    from difflib import SequenceMatcher
    import re

    refs = pd.read_json(Path('/home/chyi/hsd_utils/tests/temp_devs/hescd_refs.json'),orient='index')
    #%% updating to 09feb2026 version; adding doi columns
    ref_titles_alphanumeric_only = [re.sub(r'[^a-zA-Z0-9]', '', ref) for ref in refs['title'].to_list()]
    short_cite0=""
    full_cite0=""
    short_cite1=""
    full_cite1=""
    doi_rows = [""]*len(hsd)
    temp_titles_0212=[""]*len(hsd)
    short_cite_rows=[""]*len(hsd)
    for i, row in enumerate(hsd):
        closest_idx=-1
        short_cite1=row['short_cite']
        full_cite1=row['full citation']
        if full_cite1!=full_cite0:
            print(f"full citation: {row['full citation']}\n short_cite:{row['short_cite']}")

            try:
                closest = get_close_matches(row['full citation'], refs['title'].to_list(), cutoff=0.6)[0]
                closest_idx=refs['title'].tolist().index(closest)
                
            except IndexError as e:
                full_cite_alphanumeric_only = re.sub(r'[^a-zA-Z0-9]', '', row['full citation'])
                closest_candidates = get_close_matches(full_cite_alphanumeric_only, ref_titles_alphanumeric_only, cutoff=0.2)
                
                for j, closer in enumerate(closest_candidates):
                    closest_index = ref_titles_alphanumeric_only.index(closer)
                    print(f"{j}-th close: {refs['title'][closest_index]}")
                    print(f"while first authors (from refs): {refs['author'].tolist()[closest_index]}")
                    #print(row)
                closest_idx_in_cadidate=input("the correct index or doi: ")

                if len(closest_idx_in_cadidate)==1:
                    assert int(closest_idx_in_cadidate)<len(closest_candidates)
                    closest_idx=ref_titles_alphanumeric_only.index(closest_candidates[int(closest_idx_in_cadidate)])
                else:
                    closest_idx=refs['doi'].tolist().index(closest_idx_in_cadidate)
                
            closest_ref = refs.iloc[closest_idx]
            doi_rows[i] = closest_ref['doi']
            
            short_cite0=short_cite1
            full_cite0=full_cite1
            print("\n")
        elif short_cite1==short_cite0:
            doi_rows[i] = full_cite1
            if not full_cite0==full_cite1:
                warnings.warn(f"short citation changed from {short_cite0} to {short_cite1}, while short_cite is not changed. \n\t- full_cite0:{full_cite0}\n\t- full_cite1:{full_cite1}")
        else:
            doi_rows[i] = full_cite1

        # make a new short-cite line
        short_cite_rows[i] = closest_ref['bibitem_key']
        temp_titles_0212[i]= closest_ref['title']

    hsd._df['doi']=doi_rows
    hsd._df['short_cite_02feb']=short_cite_rows
    hsd._df['titles_02feb']=temp_titles_0212
    hsd._df.to_csv('doi_added_10feb2026.csv')