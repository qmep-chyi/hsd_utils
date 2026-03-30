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
from hsdu.dataset import Dataset
from pymatgen.core.composition import Composition
from fractions import Fraction
#print(f"{Fraction(1, 6):g}")
# Load raw dataset (26 Feb)
hsd = Dataset(r'C:\Users\chyi\hsd_utils\src\hsdu\data\tests\full_dataset.csv', exception_col='Exceptions')
hsd._df.tail(5)

print(hsd[7])