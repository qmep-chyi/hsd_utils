#%%
import argparse
from typing import Literal

import pandas as pd
from scipy.spatial.distance import cdist
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element

from hsdu.dataset import Dataset
from hsdu.utils.conversion_utils import element_list_iupac_ordered, OneHotFracCodec

def distance_matrix(comps0:list[Composition] |list[list[float]],
                    comps1:list[Composition] |list[list[float]],
                    metric:Literal["chebyshev", "cityblock", "euclidean", "L1", "l1", "L2","l2", "L_infty", "maximum", "tchebychev", "l_infty"], elemental_set:list[str]|None = None,
                    exclude_expanded_elements:bool=True):
    """wrapper of cdist from scipy.spatial.distance
    
    Args:
        - comps: list[Composition] - list of pymatgen Compositions
        - metric: str - mapping aliases,
            - "l1": to "cityblock"
                - L1 distance $\sum_i|x_i-y_i|$ = 
            - "l2": to "euclidean" 
                - L2, Euclidean distance. $\sqrt{\sum_i{(x_i-y_i})^2}$
            - "l_infty" | "maximum" | "tchebychev": to "chebyshev"
                - \\max_i(|x_i-y_i|)
        - exclude_expanded_elements:
            if True, ignore elements atomic_number>103. default True.
    """
    assert exclude_expanded_elements
    max_atomic_number=103
    comps_list=[]
    for comps in (comps0, comps1):
        if all(isinstance(comp, Composition) for comp in comps):
            if elemental_set is None:
                # load all elements symbols from pymatgen
                # instead of building a new elemental set
                elemental_set = element_list_iupac_ordered([i.symbol for i in Element][:max_atomic_number])
            onehot_frac_codec = OneHotFracCodec(elemental_set)
            comps = [onehot_frac_codec.encode(comp) for comp in comps]
            comps_list.append(comps)

        elif all(all(isinstance(frac, float) for frac in frac_row) for frac_row in comps):
            comps_list.append(comps)
        else:
            raise TypeError(comps)
        
    alias_map = {al: "chebyshev" for al in ("L_infty", "maximum", "tchebychev", "l_infty")}
    alias_map = alias_map | {al: "cityblock" for al in ("l1", "L1")}
    alias_map = alias_map | {al: "euclidean" for al in ("l2", "L2")}
    metric_out: Literal["chebyshev","cityblock","euclidean"]= alias_map.get(metric, metric)
    return cdist(comps_list[0], comps_list[1], metric=metric_out)

def comps_pymatgen2elemental_set(pymatgen_comps: Composition):
    """make elements_set list"""
    elements_sets=[]
    for comp in pymatgen_comps:
        elements_sets.append("-".join(element_list_iupac_ordered(comp.elements)))
    return elements_sets

def merge_overlap(index, duplicates_group, group_rows):
    overlaps=False
    for i in index:
        first_group=-1
        other_groups=[]
        for k, v in duplicates_group.items():
            if i in v:
                if first_group==-1:
                    first_group=k
                else:
                    other_groups.append(k)
        if len(other_groups)>0:
            overlaps=True
            for j in other_groups:
                duplicates_group[first_group].update(duplicates_group[j])
                for k in duplicates_group[j]:
                    group_rows[k]=first_group
                duplicates_group.pop(j)

    if overlaps:
        return merge_overlap(index, duplicates_group, group_rows)
    else:
        assert sum(len(set(v)) for v in duplicates_group.values()) == len(index)
        for i in index:
            assert sum([1 if i in v else 0 for v in duplicates_group.values()])==1
        return duplicates_group, group_rows
        
def make_duplicates_group(index, d1_dist_matrix, dinfty_dist_matrix, elements_sets, linfty_cutoff=0.01, l1_cutoff=0.02, cross_elements_set=False, verbose=True):
    if verbose:
        print(f"start to make duplicates group; linfty_cutoff={linfty_cutoff}, l1_cutoff={l1_cutoff}")
    assert index.tolist()==list(range(len(elements_sets)))
    last_dup_group_idx = -1 # so first is 0
    current_group_idx = 0
    duplicates_group = dict()
    group_rows = dict()

    for i in index:
        # init group
        if group_rows.get(i) is None: 
            last_dup_group_idx+=1
            duplicates_group.setdefault(last_dup_group_idx, {i})
            group_rows.setdefault(i, last_dup_group_idx)
            current_group_idx = last_dup_group_idx
        else:
            current_group_idx = group_rows[i]
        # process duplicates
        for j in index:
            if d1_dist_matrix[i, j]<=l1_cutoff and dinfty_dist_matrix[i, j]<=linfty_cutoff:
                if cross_elements_set:
                    if j<=i: # just double check
                        assert i in duplicates_group[group_rows[j]]
                    elif j>i:
                        duplicates_group[current_group_idx].add(j)
                        group_rows[j]=current_group_idx # TODO: maybe I would store multiple values to optimize merge_overlap()
                    else:
                        raise ValueError
                else:
                    if elements_sets[i]==elements_sets[j]:
                        if j<=i: # just double check
                            assert i in duplicates_group[group_rows[j]]
                        elif j>i:
                            duplicates_group[current_group_idx].add(j)
                            group_rows[j]=current_group_idx
                        else:
                            raise ValueError
                                    
    return merge_overlap(index, duplicates_group, group_rows)

if __name__=="__main__":
    from hsdu import Dataset
    # merge entries with close compositions
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help="path to the HESC dataset")
    args=parser.parse_args()

    hsd = Dataset(args.data_path)

    assert hsd.df.index.tolist()==list(range(len(hsd)))
    elements_sets = [hsd.idx2aux["elements_set"][i] for i in hsd.df.index]
    print(pd.Series(elements_sets).value_counts())

    # make distance matrix
    onehot_fracs = hsd.onehot_fracs()
    # chebyshev distance
    dinfty_dist_matrix = distance_matrix(onehot_fracs, onehot_fracs, metric="l_infty")
    # L1 distance
    d1_dist_matrix = cdist(onehot_fracs, onehot_fracs, metric="l1")

    dup_group, group_rows = make_duplicates_group(hsd.df.index, d1_dist_matrix, dinfty_dist_matrix, elements_sets, linfty_cutoff=0.01, l1_cutoff=0.02)

    print(len(dup_group))
    print(pd.Series(group_rows).value_counts())
    print([elements_sets[dup_group[i][0]] for i in pd.Series(group_rows).value_counts().index[:10]])

    


    #%% test to prevent data leakage (train-test split)

    dup_group, group_rows = make_duplicates_group(hsd.df.index, d1_dist_matrix, dinfty_dist_matrix, elements_sets, linfty_cutoff=0.05, l1_cutoff=0.12, cross_elements_set=True)

    print(len(dup_group))
    print(pd.Series(group_rows).value_counts())
