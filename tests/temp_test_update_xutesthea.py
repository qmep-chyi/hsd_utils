import importlib.resources as resources
import warnings
from pathlib import Path


from hsdu.dataset import Dataset
from hsdu.comparison import XuTestHEA
hsd=Dataset(r'...hsdu\data\dataset_20260415.csv')
xu_dataset = XuTestHEA(r'...\hsdu\data\miscs\xu2025_test_HEAs_private.csv')
xu_dataset.encode_onehot_fracs(fixed_elements_set=hsd.elemental_set, rule_elements_set='overwrite', parse_pymatgen_comps_col='formula')
dupl_group_to_xu49, idx2group_to_xu49=hsd.group_duplicates(other=xu_dataset, cityblock=0.01, msre=0.02, update_attrs=False)
dupl_group_internal, idx2group_internal=hsd.group_duplicates(cityblock=0.01, msre=0.02)
print('group_duplicates with XuTestHEA()')
existing_dupl_groups=dict()
group_indices=[]
for k, v in dupl_group_to_xu49.items():
    if len(v)>0:
        existing_dupl_groups[k]=v
        group_indices.append(k)
missing_index=[]
for i in range(len(xu_dataset)):
    if i not in group_indices:
        missing_index.append(i)
print(f'among {len(xu_dataset)} entries of XuTestHEA(), no duplicates in HE-SC dataset:{missing_index}')

similar_entries_for_df=[]
for idx, row in xu_dataset._df.iterrows():
        hesc_id = int(row['hesc_id']) # 'hesc_id' on XuTestHEA() entries! not ours.
        hsd_idx = hsd.get_idx_by_hesc_id(hesc_id)

        group_idx_internal = idx2group_internal[hsd_idx]
        group_idx_to_xu49 = idx2group_to_xu49[hsd_idx]
        group_entries_internal = [int(hsd[j]['hesc_id']) for j in dupl_group_internal[group_idx_internal]]
        if idx in missing_index:
            assert group_idx_to_xu49 is None
            warnings.warn(f'xu_dataset_idx:{idx} does not grouped with any entries on the HE-SC dataset, directly using hesc_id',UserWarning)
            similar_entries_for_df.append(str(group_entries_internal))
        else:
            group_entries_to_xu49 = [int(hsd[j]['hesc_id']) for j in dupl_group_to_xu49[group_idx_to_xu49]]
            assert group_entries_to_xu49==group_entries_internal
            similar_entries_for_df.append(str(group_entries_to_xu49))
xu_dataset._df['close_entries']=similar_entries_for_df
out_xu_test49=xu_dataset._df.drop(columns=xu_dataset.column_sets['onehot_elements'])
out_xu_test49.to_csv("xu_test_hea49_20260416.csv", index=False)
print(missing_index)
print([xu_dataset[i] for i in missing_index])
