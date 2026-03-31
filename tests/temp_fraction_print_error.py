#%% [markdown]
# # To debug TypeError on the following code
# when fractions of pymatgen_comps are `fractions.Fraction` class, cannot print `pymatgen_comps` as follows
# ```
# Exception has occurred: TypeError  
# unsupported format string passed to Fraction.__format__  
# File "C:\Users\chyi\hsd_utils\tests\temp_fraction_print_error.py", line 12, in <module>  
#    print(hsd[7])  
# TypeError: unsupported format string passed to Fraction.__format__  
# ```
#%%
from hsdu.dataset import Dataset, D2TableDataset
dataset_path = r""

test49=D2TableDataset(r'', exception_col=None, encode_onehot_fracs=False, drop_cols=['comps_pymatgen'])

# Load raw dataset (26 Feb)
hsd = Dataset(r'...\hsdu\data\tests\full_dataset.csv', exception_col='Exceptions')

test49.encode_onehot_fracs(fixed_elements_set=hsd.elemental_set, rule_elements_set='overwrite')
dupl_group, idx2group=hsd.group_duplicates(other=test49, cityblock=0.01, msre=0.02, update_attrs=False, )

train202=D2TableDataset(r'...\hsd_utils\tests\temp_devs\xgboost\HESC_train_col450.csv', exception_col=None, encode_onehot_fracs=False, drop_cols=['comps_pymatgen'])
train202.encode_onehot_fracs(fixed_elements_set=hsd.elemental_set, rule_elements_set='overwrite')
dupl_group2train202, idx2group2train202=hsd.group_duplicates(other=train202, cityblock=0.01, msre=0.02, update_attrs=False, )
