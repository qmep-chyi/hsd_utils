"""preprocess supercon dataset following the Xu 2025

see preprocess_supercon.md for descriptions

Todo:
    * hard_coded:
        * Tc_upper_bound
"""
import pandas as pd
from hsdu.comparison import StanevSuperCon, XuTestHEA
dataset = StanevSuperCon()

# filter non_SC_observed and high-Tc datapoints
Tc_upper_bound = 13.0 # hard coded value!!

df = dataset.df

valid_Tcs = []
for idx, row in df.iterrows():
    if row["Tc"] > 0.0 and row["Tc"] < Tc_upper_bound:
        valid_Tcs.append(idx)

print(f"len(valid_Tcs)/len(df):{len(valid_Tcs)}/{len(df)}")
dataset.df = dataset.df.loc[valid_Tcs].reset_index(drop=True)
assert len(valid_Tcs)==len(dataset.df)
# add comps_pymatgen (sanity check)
#   - all the comps_pymatgen should be created,
#   - because `StanevSuperCon` dropped exceptions if it cannot generate comps_pymatgen

dataset.pymatgen_comps()

duplicates = dataset.df.duplicated(subset=['comps_pymatgen'], keep=False)
print(f"duplicates.value_counts():{duplicates.value_counts()}")
dataset.df = dataset.df.drop_duplicates(subset=['comps_pymatgen'], keep=False).reset_index(drop=True)
print(f"len(dataset.dataframe):{len(dataset.df)}")

# save csv
dataset.df[["name", "Tc"]].to_csv("preprocessed_supercon.csv")