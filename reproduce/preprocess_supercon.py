# %% [markdown]
# # SuperCon Trained XGBoost
# * Using 145 features of Matminer + XGBoost Regressor
# * Raw SuperCon dataset:
#     * Center for Basic Research on Materials. (n.d.). MDR SuperCon Datasheet Ver.240322. National Institute for Materials Science. 
#     * https://doi.org/10.48505/nims.4487
#     * `20240322_MDR_OAndM.txt`: Oxides and Metallic
# * Description(PDF): https://mdr.nims.go.jp/filesets/4f1fcd20-92ea-4b53-8c2d-29281d2ca642/download
#     * *Tc* columns (in abbreviation name):
#         * `tc`---Tc (of this sample) recommended---: It is considered as a representative T_c if exist.
#         * `tc1`, `tc2`, `tc3`: Resistivity zero, 50% (mid point), 100%(onset)
#         * `tcsus`: Tc measured from susceptibility measurement
#         * `tcn`: lowest temperature for measurement (not superconducting)
#         * `tcmeth`: tc measurement method (1.magnetization, 2.ac susceptibility, 3.resistivity, 4.heat capacity, 5.tunneling, 6.infrared spectroscopy, 7.thermal conductivity, 8.Raman spectroscopy , 9.nuclear magnetic resonance, 10.surface impedance, 11.neutron diffraction, 12.photoemission spectroscopy, 13.microwave transmission, 14.Others)
#     * compositions and fractions:
#         * on the column names, ignore the number following `element name of materials`. 
#             * like `element name of materials.1`
#             * As they are just assigned to avoid duplicate column names.
#         * For Example, $Ba_{0.1}La_{1.9}Ag_{0.1}Cu_{0.9}O_{4-Y}$
#             * `ma1` (element name of materials): 'Ba'
#             * `ma2` (composition of MA1): 0.1
#             * `mb1` (element name of materials.1): 'La'
#             * `mb2` (composition of MA2): 1.9
#             * `mc1` (element name of materials.2): 'Ag'
#             * `mc2` (composition of MA3): 0.1
#             * `md1`: ...
# ## identifier and formula (string)
#   * all entry have a unique `data number`
#   * all entries have `chemical formula`
# * OLD version uses Stanev et al. file.
#     * See [preprocess_supercon.md](..\src\hsdu\data\miscs\preprocess_supercon.md)
#     * See [preprocess_supercon.py](..\src\hsdu\data\miscs\preprocess_supercon.py)

# %%
import pandas as pd
import numpy as np

import re
from warnings import warn
from pymatgen.core.periodic_table import Element
from pymatgen.core import Composition


def validate_elements(elems)->tuple[list|None, list]:
    """
    return: tuple[
            * list(elements_corrected) | None
                * None if failed to correct elements
            * list(exceptions)
        ]
    """
    exceptions=[]
    try:
        [Element(el) for el in elems]
        return elems, None
    except ValueError:
        new_elems = []
        for el in elems:
            if el.rstrip() != el:
                el = el.rstrip()
                exceptions.append('trail_white')
                
            if len(el)==2 and isinstance(el, str) and el[1].isupper():
                el=el[0]+el[1].lower()
                exceptions.append('two_upper_letter_el')
            
            new_elems.append(el)
        
        if len(new_elems)==0:
            exceptions.append('no_valid_element')
        
        if exceptions:
            return new_elems, exceptions
        else:
            return None, exceptions
        
def fix_fracs(fracs)->tuple[list[float], list[str]]:
    """
    return: tuple[list[float], list[str]]
        * fixed fractions: list[float]
        * exceptions_to_drop: list[str]
    """
    new_fracs = []
    exceptions_to_drop = [] #exceptions to drop.
    for fr in fracs:
        if isinstance(fr, str):
            try:
                new_fracs.append(float(eval(fr)))
                # eval(fraction_string) worked
            except NameError:
                variable_in_fraction = re.search(r'[XxYyZzDd]', fr)
                if variable_in_fraction is not None:
                    # checkout only if it is a valid expression
                    eval_variable = re.sub(r'[XxYyZzDd]', '0.1', fr)
                    try:
                        float(eval(eval_variable))
                        exceptions_to_drop.append('equation_like_valid')
                        break
                    except ValueError:
                        exceptions_to_drop.append('equation_like_invalid')
                        break
                else:
                    try:
                        new_fracs.append(float(fr))
                    except ValueError:
                        exceptions_to_drop.append('NameError_float(fr)')
                        warn(f'cannot parse {i}: (NameError) {row[comp_cols]}', UserWarning)
                        break
            except TypeError as et:
                if not exceptions_to_drop and re.search(r'(?<=\d),(?=\d)', fr):
                    new_fracs.append(float(fr.replace(',','.'))) # comma -> period
                else:
                    raise et
            except SyntaxError:
                exceptions_to_drop.append('SyntaxError')
                warn(f'cannot parse {i} (SyntaxError): {row[comp_cols]}', UserWarning)
                break
        else:
            new_fracs.append(fr)
    
    non_zero_fracs = [fr for fr in fracs if fr!=0]
    assert len(new_fracs)==len(non_zero_fracs) or exceptions_to_drop
    return new_fracs, exceptions_to_drop

def validate_fracs(elems, fracs)->tuple[Composition | None, list]:
    """
    return: 
        * pymatgen_comps: Composition() object(from pymatgen.core).
        * exceptions_to_drop: log, reason to drop. notes.
    """
    exceptions_to_drop=[]

    try:
        comps={el:float(fr) for el, fr in zip(elems, fracs) if pd.notna(el)}
        pymatgen_comps=Composition(comps)

    except ValueError:
        new_fracs, exceptions_fracs =fix_fracs(fracs)
                
        if exceptions_fracs:
            pymatgen_comps=None
            warn(f'cannot parse {i} fraction:{fracs}',UserWarning)
            exceptions_to_drop.append('invalid_frac')          
        
        else:
            try:
                comps={el:float(fr) for el, fr in zip(elems, new_fracs) if pd.notna(el)}
                pymatgen_comps=Composition(comps)
            except ValueError:
                pymatgen_comps=None
                exceptions_to_drop.append('others')
                warn(f'cannot parse {i}:{row[comp_cols]}', UserWarning)
    if not exceptions_to_drop and len(elems)!=len(pymatgen_comps):
        exceptions_to_drop.append(f'len(elems):len({elems})!=len(pymatgen_comps):len({pymatgen_comps})')
        warn(f'index {i}: len(elems):len({elems})!=len(pymatgen_comps):len({pymatgen_comps}', UserWarning)
        return None, exceptions_to_drop
    else:
        return pymatgen_comps, exceptions_to_drop

def elements_set_key(comp: Composition, sep: str='-')->str:
        """ return `elements-set-key` string of the comp for grouping,
        unique for the set of constituent elements (regardless of stoichiometry)
        
        args:
            * comp: Composition from pymatgen.core
            * sep: separator between elemental symbols. default '-'.
            
        return: 
            * elements-set-key, element symbols are ordered in IUPAC order.
        """
        elements = comp.elements
        sorted_elems = sorted(elements, key=lambda x: x.iupac_ordering)
        sorted_symbols = [el.symbol for el in sorted_elems]
        return sep.join(sorted_symbols)
#%%
if __name__=="__main__":
    # 20240322 MDR Oxides and M
    print("loads raw data '20240322_MDR_OAndM.txt' as as a pd.DataFrame")
    
    #your path to hsdu pacakge or 20240322_MDR_OAndM.txt from MDR SuperCon
    #### replace it to yours ####
    supercon_raw = r'$(path_to_hsdu_pacakge)\src\hsdu\data\miscs\20240322_MDR_OAndM.txt' 
    data=pd.read_csv(supercon_raw, sep='\t')

    # drop full column names, use abbreviations.
    col_abbreviation=data.loc[0]
    data = data.rename(columns={k:v for k, v in col_abbreviation.items()})
    data.drop(index=0, inplace=True)

    # %% [markdown]
    # # Preview the Dataset
    # * Among the columns seems to be useful,
    # * examine how many values available.

    # %%
    composition_cols = ['element name of materials', 'composition of MA1', 'element name of materials.1', 'composition of MA2', 'element name of materials.2', 'composition of MA3', 'element name of materials.3', 'composition of MA4', 'element name of materials.4', 'composition of MA5', 'element name of materials.5', 'composition of MA6', 'element name of materials.6', 'composition of MA7', 'element name of materials.7', 'composition of MA8', 'element name of materials.8', 'composition of MA9', 'element name of materials.9', 'composition of MA10', 
                        'oxygen', 'common formula of oxygen', 'measured value of Oxygen content']
    comp_cols = [col_abbreviation[k] for k in composition_cols]
    name_cols = ['data number', 'reference number', 'comment', 'common formula of materials', 'chemical formula', 'common name of structure']
    name_cols = [col_abbreviation[k] for k in name_cols]

    # tc_values:
    tc_values=['transition temperature (R = 0)', 'transition temperature (mid point)', 'transition temperature (R = 100%)', 'Tc from susceptibility measurement', 'Tc (of this sample) recommended']
    tc_values=[col_abbreviation[k] for k in tc_values]
    tc_misc = ['unit of Tc', 'figure number for Tc(p, x, etc)', 'tc measurement method', 'transition width for resistive transition', 'lowest temperature for measurement (not superconducting)']
    tc_misc = [col_abbreviation[k] for k in tc_misc]
    
    # %%
    print("##### Checkout available (not missing) values: #####")
    print(data[comp_cols].describe())
    # %% [markdown]
    # ## Tc values
    # Among 33458 entries--
    # * `tc`: 26358
    # * `tc2`: 17117
    # * `tcsus`: 5624
    # * `tc1`, `tc3`: only 4051 and 4060 entries have, among 33458 entries.
    # * `tcn`: 4641 (lowest temperature for measurment)
    # * `utc` (unit of Tc): 24166 `K` and 1 `k`. just typo
    # * others too few

    # %%
    data[tc_values+tc_misc].describe()

    # %%
    data['utc'].value_counts()

    # %%
    # the case unit of Tc=`k`
    print(f"{data[data['utc']=='k'][tc_values+tc_misc]=}")
    

    # %% [markdown]
    # # Validate Tc values
    # * preprocess: if `tcn` or `tc` values are 0, remove. It cannot be zero.
    # * exceptional cases:
    #     1. If both `tcn` and valid `tc` exist
    #         1. if `tcn`>=`tc`, it might be the onset temperature
    #         1. if `tcn`<`tc`, it might be the baes temperature, lowest temperature reached on the measurment.
    #     1. If both `tcn` and valid `tc` missing: It is useless!

    # %%
    print("start processing Tc value")
    # remove all tcn==0
    print(f"remove {len(data.loc[data['tcn']==0])} tcn values that was 0.")
    data.loc[data['tcn']==0, 'tcn']=np.nan
    assert len(data.loc[data['tcn']==0])==0

    # %%
    # zero tc values to be removed(it cannot be zero)
    for tc_col in tc_values:
        print(f'removing 0 tc in {tc_col=} (number of zero tc values: {len(data.loc[data[tc_col]==0])})')
        data.loc[data[tc_col]==0, tc_col]=np.nan
        assert len(data.loc[data[tc_col]==0])==0

    # %%
    #count_rep_tc = [] # count representative Tc, 'tc'

    # list indices of exceptional cases.
    both_tcn_and_tc=[]
    no_tcn_no_tc=[]
    tcn_larger_than_tc=[]

    for i, row in data.iterrows():
        valid_tcs = row[tc_values].apply(float).tolist()
        tcn=float(row['tcn'])
        valid_tcs = [t for t in valid_tcs if t is not None and not pd.isna(t)]
        
        # check 1. validation of 'tcn'
        
        # both valid tcn and valid_tcs exist
        if not pd.isna(tcn) and len(valid_tcs)!=0: 
            if tcn<=min(valid_tcs):
                # can be the lowest (base) temperature of the measurement (as the metadata of the measurement)
                both_tcn_and_tc.append(i)
            else: # tcn>min(valid_tcs) case - requires some explanation
                tcn_larger_than_tc.append(i)
        elif len(valid_tcs)==0:
            pass # expected common case: if tcn exist, no valid tcs
        elif pd.isna(tcn) and len(valid_tcs)>0:
            pass # expected common case: have valid tcs, no tcn
        else:
            raise ValueError

        # check 2. no tcn, no valid tc.
        if pd.isna(tcn) and len(valid_tcs)==0:
            no_tcn_no_tc.append(i)
            # in this case, no information we can get.
    
    print("irregular cases on Tc values:")
    print({
        'both_tcn_and_tc count':len(both_tcn_and_tc), 
        'tcn_larger_than_tc count':len(tcn_larger_than_tc),
        'no_tcn_no_tc count':len(no_tcn_no_tc)
        })

    # %%
    print(f"{data.loc[both_tcn_and_tc[-1]][tc_values+['tcn']]=}")

    # %%
    # Exceptions: 'tcn' > tc:
    #   * `tcn` might be the onset temperature where transition started, mistakenly.
    print(f"{data.loc[tcn_larger_than_tc][tc_values+['tcn']]=}")

    # %%
    print("##### remove all these exceptions (as a strict, rigorous case) #####")
    exception_idx = set(both_tcn_and_tc) | set(tcn_larger_than_tc) | set(no_tcn_no_tc)
    print(f"{len(exception_idx)=} entries will be dropped")
    #len(set(both_tcn_and_tc).(tcn_larger_than_tc).add(no_tcn_no_tc))
    df = data.drop(index=exception_idx)

    # %% [markdown]
    # ### examine measured tc values
    # #### Mostly, entries have `tc`.
    # * if SC is observed (i.e. do not have a valid `tcn`)
    # * without exceptions dropped above.
    # * Dropped 7 cases without 'tc' and 'tcn', only have 'tc3'

    # %%
    print("##### preview tc values #####")
    # 26205/30307 entires have valid 'tc'
    df['tc'].isna().value_counts()

    # %%
    # 4095/30307 entires have valid 'tcn'
    df['tcn'].notna().value_counts()

    # %%
    # now, every entry has 'tcn' or 'measured Tc value' (never both, never neither)
    assert len(df[df['tcn'].notna() & df[tc_values].notna().any(axis=1)])==0

    # %% [markdown]
    # #### Entries only with tc3 (onset)
    # * `tc3`: R=100% temperature during the SC transition (onset of the transition)

    # %%
    # most entries have 'tc' values. these 7 entries do not. Possibly only the 'onset' of the transition is observed.
    df[df['tcn'].isna() & df['tc'].isna()][tc_values+['tcn']]

    # %%
    # entries with no measured Tc value except for the 't3' (in result, same entries with above!)
    tcs_without_t3 = tc_values.copy()
    tcs_without_t3.remove('t3')
    df[df['t3'].notna() & df[tcs_without_t3].isna().all(axis=1)][tc_values+tc_misc]

    # %%
    # entries with valid 'tc' value but 't3'='tc', (and do not have 'tcsus')
    df[df['t1'].isna() & df['t2'].isna() & df['t3'].notna() & ((df['tc']==df['t3']) & (df['tcsus'].isna()))][tc_values]

    # %% [markdown]
    # #### `tcsus`: Tc by magnetic susceptibility

    # %%
    # every entry measured `tcsus` have `tc`
    assert len(df[df['tcsus'].notna() & df['tc'].isna()])==0
    # available 'tcsus'
    df[df['tcsus'].notna()][tc_values]

    # %%
    # both 'tc' <-> 'tcsus' available
    print("Entries have both 'tc' and 'tcsus' available and `tc!=tcsus`:")
    print(df[df['tc'].notna() & df['tcsus'].notna() & (df['tc']!=df['tcsus'])][tc_values])
    print("`tc==tcsus` entries:")
    print(df[df['tc'].notna() & df['tcsus'].notna() & (df['tc']==df['tcsus'])][tc_values])

    # %%
    # drop entries without 'tc' and 'tcn' but 't3' exist
    print("###### drop entries without 'tc' and 'tcn' but 't3' exist #####")
    df_tc_cleaned=df.drop(index=df[df['tc'].isna() & df['tcn'].isna() & df['t3']].index)
    #df = df.drop([df['tc'].isna() & df['tcn'].isna() & df['t3']])
    df_tc_cleaned

    # %% [markdown]
    # ### Result (Preprocess Tc values)
    # * Clean Tc values: Dropped 3151 rows, 30307 entries left.
    #     1. zero `tcn`, zero tc values are removed:
    #         * 483 `tcn`, 
    #         * `t1`: 17, `t2`: 40, `t3`: 19, `tcsus`: 25, `tc`: 90
    #     1. rows with exceptions are dropped
    #         * 'both_tcn_and_tc count': 51,
    #         * 'tcn_larger_than_tc count': 12,
    #         * 'no_tcn_no_tc count': 3088
    #     1. Finally, 7 entries only with `t3` removed.
    #         * Any of `tc`, `t1`, `t2`, `tcsus` and `tcn` were valid on these entries
    #         * considering only the 'onset' of the transition is observed.
    #         * by the way, there are 1731 entries
    #             * of which `tcn` is equal to `t3`
    #             * no other tc values available (at the same time)
    #             * So, it might be an arbitrary choice.
    # * tc_cleaned table:
    #     * 26205 entries have `tc`---Tc (of this sample) recommended---
    #     * 4095/30307 entires with valid `tcn`: 
    #         * considered as a `non-SC observed`,
    #         * possibly `T_c` exist but  lower than its 'tcn'
    #     * if a valid `tcn` exist, there is no valid tc values. and vice versa---if 'tc' exist, no 'tcn'---
    #     * `tcsus` is valid on 5593 entries

    # %%
    # every rows have only a 'tcn' or a 'tc' value.
    for i, row in df_tc_cleaned.iterrows():
        valid_tcs = row[tc_values].apply(float).tolist()
        tcn=float(row['tcn'])
        valid_tcs = [t for t in valid_tcs if pd.notna(t)]
        if pd.isna(tcn):
            assert len(valid_tcs)>0
        else:
            assert len(valid_tcs)==0

    # %% [markdown]
    # # Validate Compositions

    # %%
    #elements / fraction columns:
    elem_cols = [f'm{i}1' for i in 'abcdefghij']+['mo1']
    frac_cols = [f'm{i}2' for i in 'abcdefghij']+['mo2']
    # For oxygen column, 'mo1', 'oz' might be useful
    elem_cols, frac_cols

    # %% [markdown]
    # ## Parse, Compile Compositions
    # * from element and fraction columns:
    #     * Elements from ['ma1', 'mb1', 'mc1',...,'mh1']. (element name of materials) Element symbols.
    #     * Fractions from ['ma2',...,'mh2']. (composition of MA1-MA10)
    # * Some strings failed to parse are auto-corrected:
    #     * Element string
    #         * have trailing whitespace.
    #         * Uppercase seconde letter 
    #     * Fraction have comma: `3,4`->`3.4`


    # %% [markdown]
    # #### IUPAC ordering:
    # * sort symbols by IUPAC ordering
    # * see `pymatgen.core.periodic_table.ElementBase.iupac_ordering` for details
    # * `elements_set_key` function returns elements set in iupac ordering.
    # %%
    elements_symbols = [i.symbol for i in Element]

    pymatgen_comps = dict() # dict[index, Composition]
    rows_to_drop = dict() # dict[index, list[str]], reason why drop this row.
    elements_set_keys = dict() # dict[index, str]. elements-set, IUPAC ordered.

    drop_rows=dict() # exceptions that cannot be auto-corrected

    for i, row in data.iterrows():
        elems0 = [row.get(k) for k in elem_cols]
        elems = [el for i, el in enumerate(elems0) if pd.notna(elems0[i])]

        pymatgen_comps[i]=None

        new_elems, elem_exceptions = validate_elements(elems=elems)
        drop_rows[i]=[]
        if new_elems is None:
            drop_rows[i].append(f'invalid_elements: {elems}')
            warn(f'cannot parse elements: {elems}', UserWarning)
            continue
        else:
            elems=new_elems
        
        fracs0 = [row.get(k) for k in frac_cols]

        #fracs1 = [fr for i, fr in enumerate(fracs0) if pd.notna(elems0[i])]
        #fracs = [fr if pd.notna(fracs1[i]) else 0.0 for i, fr in enumerate(fracs1)]
        fracs = [fr if pd.notna(fr) else 0.0 for fr, el in zip(fracs0, elems0) if pd.notna(el)]

        if not drop_rows[i]:
            # if not decided to drop yet
            pymatgen_comps[i], frac_exceptions = validate_fracs(elems, fracs)

            if frac_exceptions:
                drop_rows[i].append(f'invalid_composition: {fracs}')
            else:
                elements_set_keys[i]=elements_set_key(pymatgen_comps[i])

    # %% [markdown]
    # #### 'mo1'는 O라고 해놨을 뿐 O가 없었던 경우는 없을까?
    # - 'mo1'은 있지만 'mo2', 'oz'는 없는 경우:
    # - 모두 element, name에 Oxygen을 적시하였음.

    # %%
    data[data['mo1'].notna() & data['mo2'].isna() & data['oz'].isna()][['element', 'mo1', 'mo2', 'oz']]

    # %% [markdown]
    # ### final check

    # %%
    print("##### Final Check (before group duplicates) #####")
    # Every entry has exactly one of the Composition and Drop_rows (mutually exclusive)
    assert len(drop_rows)==len(pymatgen_comps)
    pymatgen_comps_not_none = {k:v for k, v in pymatgen_comps.items() if v is not None}
    drop_rows_not_none = {k:v for k, v in drop_rows.items() if v}
    assert len(set(pymatgen_comps_not_none.keys()))+len(set(drop_rows_not_none.keys()))==len(data)
    assert len(set(pymatgen_comps_not_none.keys()) & set(drop_rows_not_none.keys()))==0

    # %% [markdown]
    # ## Drop rows with invalid composition

    # %%
    comps_df = pd.DataFrame.from_dict(dict(comps_pymatgen=pymatgen_comps, row_to_drop=drop_rows_not_none, elements_set_key=elements_set_keys))
    assert len(comps_df[comps_df['comps_pymatgen'].isna() & comps_df['row_to_drop'].isna()])==0
    assert len(comps_df[comps_df['comps_pymatgen'].notna() & comps_df['row_to_drop'].notna()])==0
    assert len(comps_df[comps_df['comps_pymatgen'].notna() & comps_df['elements_set_key'].isna()])==0

    # %%
    comps_df['row_to_drop']

    # %%
    count_invalid=0
    for index, row in comps_df.iterrows():
        if pd.notna(row['row_to_drop']):
            if any(['invalid' in e for e in row['row_to_drop']]):
                count_invalid+=1
    print(f"{count_invalid=}")

    # %%
    pd.notna(comps_df['row_to_drop']).value_counts()
    # %%
    comps_df = comps_df[['comps_pymatgen', 'elements_set_key']].dropna()
    comps_df

    # %% [markdown]
    # # Tc, composition cleaned df
    # * have valid-
    #     * Composition (from pymatgen.core)
    #     * `tc` (float)

    # %%
    df_cleaned = df_tc_cleaned.join(comps_df, how='inner')
    df_cleaned

    # %%
    # have pymatgen composition
    assert len(df_cleaned[df_cleaned['comps_pymatgen'].isna()])==0
    assert all(df_cleaned[df_cleaned['comps_pymatgen'].apply(lambda x: len(x))==0]==0) #of which length!=0
    # 'tc', 'tcn': mutually exclusive
    assert len(df_cleaned[df_cleaned['tc'].isna() & df_cleaned['tcn'].isna()])==0
    assert len(df_cleaned[df_cleaned['tc'].notna() & df_cleaned['tcn'].notna()])==0
    # pymatgen Composition length == numbers of elements in elements_set_key.
    assert all(df_cleaned[df_cleaned['comps_pymatgen'].apply(len) == df_cleaned['elements_set_key'].apply(lambda x: x.split('-'))])


    # %% [markdown]
    # # Duplicates
    # * ***Data Redundancy*** On [Xu et al., 2025](10.15302/frontphys.2025.014205)
    #     > Firstly, the identical composition has different Tc values.  
    #     > Secondly, the proportions of elements in the material are adjusted in multiples, or the order of elements is modified.  
    #     > As a result, we cleaned out 6505 data points of this type.  
    # * Ambiguous criteria: 그래서 중복 값들을 어떻게 처리했는지 논문에서 밝히지 않고 있습니다.
    #     * 지금까지 HE-SC 프로젝트에서는 max_Tc를 남겨서 target으로 썼습니다.

    # %% [markdown]
    # ## Normalize sum(fractions)
    # * additional step for validation, as most functions already working with `Composition.get_atomic fractions()`.
    # * make sum(fractions) to be 1.0

    # %% [markdown]
    # ```python
    # # from hsdu.dataset.D2TableDataset.normalize_fractions
    # def normalize_fractions(compositions:dict[int, Composition], invalid_frac='keep'):
    #     out_compositions=dict()
    #     for idx, comp in compositions.items():
    #         if comp is not None:
    #             fracs=[comp[el] for el in comp.elements]
    #             if all([fr is not None for fr in fracs]):
    #                 out_comp=Composition({el:comp.get_atomic_fraction(el) for el in comp.elements})
    #                 if np.allclose(sum([out_comp[el] for el in out_comp.elements]), 1.0):
    #                     out_compositions[idx]=out_comp
    #                 else:
    #                     raise ValueError(idx, comp)
    #             elif invalid_frac is None:
    #                 raise ValueError(fracs)
    #             elif invalid_frac=='keep':
    #                 raise ValueError(comp)
    #             else:
    #                 raise NotImplementedError(invalid_frac)
    #         elif invalid_frac=='keep':
    #             out_compositions[idx]=None
    #         else:
    #             raise ValueError(comp)
    # 
    #     assert compositions.keys()==out_compositions.keys()
    # 
    #     return compositions
    # 
    # comps_norm = normalize_fractions(df_cleaned['comps_pymatgen'].to_dict(), invalid_frac=None)
    # comps_norm
    # ```

    # %% [markdown]
    # ## 비슷한 화학식 묶기 (자체구현 코드)
    # * We will use in-house code
    #     * `distance_matrix`: return distance matrices, wrapper of scipy.cdist
    #     * `make_duplicates_group`: according to the distance, group entries.
    #         * it is unoptimized code, as I have been working with HE-SC dataset have <500 entries
    # * Using `hsdu.dataset.D2TableDataset.group_duplicates`
    #     * Note: I plan to refactor this because it is too complex
    #         * for the readibility and the ease of maintenance.
    #         * Nowadays, I found that a rather long example code is not that bad, in many ways.
    #         * So I have tried not to use `Dataset` classes and not to refactor these codes on such modules but it was too hard.

    # %%
    from hsdu.utils.duplicate import make_duplicates_group, distance_matrix
    from hsdu.dataset import D2TableDataset
    # %%
    print("##### Group Duplicates #####")
    df_cleaned['supercon_index']=df_cleaned.index.to_series()
    df_cleaned

    # %% [markdown]
    # ### D2TableDataset
    # * duplicate composition을 검출하기 위해,
    # * 자체구현 class인, D2TableDataset을 사용합니다.
    # * 'tc'가 없는 entry들은 모두 제거합니다.
    #     * 우리 연구에서 아직 non-SC observed entry를 쓰지 않았고,
    #     * Xu et al. 2025에서도 사용하지 않았습니다.

    # %%
    df_cleaned=df_cleaned[df_cleaned['tc'].notna()]
    dataset = D2TableDataset(df_cleaned, exception_col=None, encode_onehot_fracs=False)
    dataset.encode_onehot_fracs(inplace=True, composition_col=df_cleaned['comps_pymatgen'])
    dataset._df

    # %%
    # 미리 현재 dataset의 index에서 원래 SuperCon 데이터프레임(data)의 index를 mapping해 둡니다.
    #   * note: 이 index는 id(식별자)가 아니라 불러온 파일에서의 순서입니다.\
    #       SuperCon에서 부여한 id는 'num'인데, 쓰지 않고 있습니다.

    #TODO: refactor. using this because the 'index_col' argument of 'D2TableDataset' is not working as expected
    idx_to_supercon_idx = {idx:supercon_idx for idx, supercon_idx in dataset._df['supercon_index'].items()}
    supercon_idx_to_idx = {v:k for k, v in idx_to_supercon_idx.items()}

    # %% [markdown]
    # #### 추가 Validation: onehot-elements와 Composition.to_pretty_string()
    # * 이 데이터셋 객체는 elements-fraction `onehot-fracs`라는 자체제작 representation으로 저장합니다.
    #     * 먼저 데이터셋에 포함된 모든 원소에 대해 column을 만들고,
    #     * sum=1이 되도록 atomic fraction을 계산해서 값으로 집어넣습니다. 
    # * `pymatgen`의 Composition 객체는 to_pretty_string() 메서드를 지원하는데

    # %%
    for i, row in dataset._df.iterrows():
        assert np.allclose(dataset.onehot_codec.encode(Composition(row['comps_pymatgen'].to_pretty_string())), row[dataset.column_sets['onehot_elements']].tolist())

    # %% [markdown]
    # ### group by dist: L1=1e-5
    # * results:
    #     * groups have only one entry: 10460
    #     * number of groups: 13_429
    #     * from `21_214` entries (Tc, composition cleaned)

    # %% [markdown]
    # #### method: group_duplicates 
    #    * 이 method는 주어진 metric에 따라 composition 사이의 거리를 계산합니다.
    #    * Return: tuple[dict, dict]
    #        * dup_group (duplicate_group): dictionary, mapping - group_index->list[그룹에 속한 entry들의 index]
    #        * idx_to_group: dictionary, mapping index-> group_index
    # 

    # %%
    # take some minutes (about 3 minutes on Ryzen 5 5600, 32GB ram)
    #dup_group, idx_to_group = dataset.group_duplicates(cityblock=1e-5)
    dup_group, idx_to_group = dataset.group_duplicates(cityblock=0.01, smape=0.02)

    # %% [markdown]
    # ## Analyze groups
    # * 그룹 분석을 위해 dup_group에서 새 dataframe을 만듭니다.

    # %%
    group_lengths={k: len(v) for k, v in dup_group.items()}
    groups_df = pd.DataFrame.from_dict(group_lengths, orient='index', columns=['size'])
    groups_df['members']=dup_group
    groups_df

    # %%
    # 중복 composition이 없는 size=1 그룹이 10190개로 가장 많습니다.(cityblock=1e-5)
    # 7164 size-1 groups (cityblock=0.01, smape=0.02)
    groups_df['size'].value_counts().head(10)

    # %% [markdown]
    # ### 그룹 내 Tc 분석

    # %%
    # dataset 에서 tc 값들을 가져와서 groups_df['tcs']에 list로 넣습니다.
    group_tcs = [[dataset._df.loc[i, 'tc'] for i in row['members']] for _, row in groups_df.iterrows()]
    groups_df['tcs']=group_tcs
    groups_df

    # %%
    # 각종 stat들을 계산합니다.
    groups_df['min']=groups_df['tcs'].apply(lambda x: min([float(i) for i in x]))
    groups_df['max']=groups_df['tcs'].apply(lambda x: max([float(i) for i in x]))
    groups_df['mean']=groups_df['tcs'].apply(lambda x: np.mean([float(i) for i in x]))
    groups_df['std']=groups_df['tcs'].apply(lambda x: np.std([float(i) for i in x]))
    groups_df['cv']=groups_df['std']/groups_df['mean']
    groups_df

    # %%
    cv_rank20 = groups_df['cv'].sort_values(ascending=False)[:20]
    cv_rank20.to_dict()

    # %% [markdown]
    # ### High coefficient of variation groups:(group criteria: cityblock=1e-5)
    # * 요약: 'Pressure dependancy'가 많지만, sample form 등 다른 이유도 많다.
    # * rank10 entries:
    #     1. Pt: 0.03K~0.00062K. 높게 나온 entry 논문 title:'Tc-enhancement in superconducting granular platinum'
    #     1. SrTiO3: Nb-doped 한 entry가 높게 나온 걸로 보임 (Nb 분율 표기 따로 안돼있음)
    #     1. Hf1Mo2: 알 수 없음. 모두 1970년대 결과. {17902: 1.0, 17903: 0.076, 17904: 0.065, 19549: 0.05, 23589: 0.07}
    #     1. MoTe2: 0.1-0.3 나오다가 P=1.1GPa, 11.7GPa일 때 6.1, 8.1
    #     1. C: diamond 등 여러 구조가 섞여있고, dopping된 element 빠진 경우가 보임. 
    #     1. EuBiS2F: P=2.4GPa
    #     1. AsNb3: 압력으로 structural transition
    #     1. La1Fe1P1F0.1O0.9: (title) Systematic Study on Fluorine-Doping Dependence of Superconducting and Normal State Properties in LaFePO1-xFx
    #     1. ...
    #     1. (rank12) Cd3As2: thin films(32989, 32990, 32991), P-induced SC(32992), Proximity-induced surface SC(32242)
    #         * `shape` 항목이 있지만 대체로 비워놔서 알 수 없다. 전체 데이터셋중 `shape`값이 있는 entry는 50% 이하 (15969/33458)
    #     1. (rank13) FeSe: 'single-layer flim' Tc 증가
    # * 노트: coefficient of variation: std/mean

    # %%
    # cv가 높은 group들에 속한 entry들의 index, tc
    top_cvs={f'rank{j+1}':{idx_to_supercon_idx[i]:data.loc[idx_to_supercon_idx[i], 'tc'] for i in groups_df.loc[cv_rank20.index.tolist()[j], 'members']} for j in range(10)}
    top_cvs

    # %%
    # rank1
    {i:data.loc[i].dropna().to_dict() for i in top_cvs['rank1']}

    # %% [markdown]
    # ### cv 시각화 (역 누적합계)
    # * $\textbf{cv}<=0.2$ 기준으로 하면 1769 entry를 버리게 됩니다. (group criteria: cityblock=1e-5)

    # %%
    dataset._df['duplicates_group']=idx_to_group
    dataset._df['cv']=[groups_df.loc[group_idx, 'cv'] for _, group_idx in idx_to_group.items()]
    dataset._df

    # %%
    import plotly.express as px
    fig=px.ecdf(dataset._df.dropna(subset='cv'), x='cv', ecdfmode='complementary', ecdfnorm=None, markers=True)
    fig.write_image('inverseCDF.png')

    # %% [markdown]
    # # 그룹 내 tc 편차가 큰 그룹 처리:
    # * 'shape'가 bulk인 경우(1, 2, 3)를 다시 group지어 봅니다. 
    # * Note: SuperCon 문서에서
    #     >shape: sample form (1: single phase(bulk),2: multi phase(bulk),3: single crystal(bulk) ,4:film,5:film(single))

    # %%
    shapes = [[dataset._df.loc[i, 'shape'] for i in row['members']] for _, row in groups_df.iterrows()]
    group_tcs = [[dataset._df.loc[i, 'tc'] for i in row['members']] for _, row in groups_df.iterrows()]

    groups_df['bulk_tcs']=[[tc for tc, s in zip(tc_list, shape_list) if s in (1, 2, 3)] for tc_list, shape_list in zip(group_tcs, shapes)]
    groups_bulk_df = groups_df.loc[groups_df['bulk_tcs'].apply(lambda x: len(x)>0)]['bulk_tcs'].copy().to_frame()
    groups_df = groups_df.drop(columns=['bulk_tcs'])

    # %%
    groups_bulk_df['max_bulk']=groups_bulk_df['bulk_tcs'].apply(lambda x: np.max([float(i) for i in x]))
    groups_bulk_df['min_bulk']=groups_bulk_df['bulk_tcs'].apply(lambda x: np.min([float(i) for i in x]))

    groups_bulk_df['mean_bulk']=groups_bulk_df['bulk_tcs'].apply(lambda x: np.mean([float(i) for i in x]))
    groups_bulk_df['std_bulk']=groups_bulk_df['bulk_tcs'].apply(lambda x: np.std([float(i) for i in x]))
    groups_bulk_df['cv_bulk']=groups_bulk_df['std_bulk']/groups_bulk_df['mean_bulk']
    groups_bulk_df

    # %% [markdown]
    # ## 그룹 내 명시적으로 Bulk인 것들의 통계

    # %%
    groups_df_merged=pd.concat([groups_df, groups_bulk_df], axis=1, join="outer")
    assert all(groups_df_merged.index==groups_df.index)
    groups_df_merged

    # %% [markdown]
    # ## Dataset에 cv_bulk 반영, 분석
    # * For example, there are 506 rows of which $cv>0.2$ and $group\_cv<0.2$ 

    # %%
    dataset._df['cv_bulk']=[groups_df_merged.loc[group_idx, 'cv_bulk'] for _, group_idx in idx_to_group.items()]
    dataset._df

    # %%
    dataset._df[(dataset._df['cv']>0.2) & (dataset._df['cv_bulk']<0.2)].sort_values(by='cv_bulk', ascending=False)

    # %% [markdown]
    # ## Export preprocessed_supercon.csv
    # * 'tc'로 어떤 값을 쓸지, 어떻게 합칠지 특별히 기준을 찾을 수 없었습니다.
    # * 그러므로, 산출한 stat들을 모두 export하도록 합니다.
    # * composition: pymatgen Composition의 `to_pretty_string` 메서드를 이용합니다.

    # %%
    group_pretty_string_df = pd.DataFrame.from_dict({index:dataset._df.loc[members[0], 'comps_pymatgen'].to_pretty_string() for index, members in groups_df_merged['members'].items()}, orient='index', columns=['composition',])
    assert all(group_pretty_string_df.index==groups_df_merged.index)
    group_pretty_string_df

    # %% [markdown]
    # ### 쓸만한 정보들을 조금 추가하고 export 합니다.

    # %%
    out_df = pd.concat([groups_df_merged, group_pretty_string_df], axis=1)
    out_df['supercon_num'] = [[data.loc[idx_to_supercon_idx[i],'num'] for i in members] for members in groups_df_merged['members']]
    rename_dicts = {old:old+'_tcs' for old in ['min', 'max', 'mean', 'std', 'cv', 'max_bulk', 'min_bulk', 'mean_bulk', 'std_bulk', 'cv_bulk']}
    rename_dicts['size']='group_size'
    out_df=out_df.rename(columns=rename_dicts)

    # columns order:
    out_df=out_df[['group_size', 'composition', 'supercon_num', 'tcs', 'min_tcs', 'max_tcs', 'mean_tcs', 'std_tcs',
                'cv_tcs', 'bulk_tcs', 'max_bulk_tcs', 'min_bulk_tcs', 'mean_bulk_tcs',
                'std_bulk_tcs', 'cv_bulk_tcs']]
    out_df.to_csv('preprocessed_supercon_20260513.csv', index_label='group_id')


    # %% [markdown]
    # ## Featurize:
    #%%
    comps_pymatgen_df=pd.DataFrame()
    comps_pymatgen_df['comps_pymatgen'] = {index:dataset.idx2aux['comps_pymatgen'][index] for index, members in groups_df_merged['members'].items()}
    out_df = pd.concat([out_df, comps_pymatgen_df], axis=1)
    # %% featurization
    from matminer.featurizers.base import MultipleFeaturizer
    from hsdu.preprocess.utils import featurizer_config_loader
    dataset = D2TableDataset(out_df, exception_col=None, encode_onehot_fracs=False)
    featurized_df = pd.DataFrame()
    featurized_df['comps_pymatgen']=comps_pymatgen_df['comps_pymatgen']

    featurizers_list, col_names_df = featurizer_config_loader(config='comp450', override_njobs=False)
    featurizer = MultipleFeaturizer(featurizers_list)
    featurizer.featurize_dataframe(featurized_df, col_id='comps_pymatgen', inplace=True)

    # comps_pymatgen column is a Composition object so drop or get string.
    featurized_df['comps_pymatgen']=featurized_df['comps_pymatgen'].apply(lambda x:x.to_pretty_string())
    #%%
    featurized_df.to_csv('supercon_maxTc_comp450.csv')

    #%%
        # %% featurization
    from matminer.featurizers.base import MultipleFeaturizer
    from hsdu.preprocess.utils import featurizer_config_loader
    dataset = D2TableDataset(out_df, exception_col=None, encode_onehot_fracs=False)
    featurized_df = pd.DataFrame()
    featurized_df['comps_pymatgen']=comps_pymatgen_df['comps_pymatgen']

    featurizers_list, col_names_df = featurizer_config_loader(config='comp146', override_njobs=False)
    featurizer = MultipleFeaturizer(featurizers_list)
    featurizer.featurize_dataframe(featurized_df, col_id='comps_pymatgen', inplace=True)

    # comps_pymatgen column is a Composition object so drop or get string.
    featurized_df['comps_pymatgen']=featurized_df['comps_pymatgen'].apply(lambda x:x.to_pretty_string())
    featurized_df.to_csv('supercon_maxTc_comp146.csv')
