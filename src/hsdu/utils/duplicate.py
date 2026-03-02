#%%
import argparse
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element

from hsdu.dataset import Dataset
from hsdu.utils.conversion_utils import element_list_iupac_ordered, OneHotFracCodec

def max_relative_error(xa, xb, symmetric=True):
    assert symmetric, NotImplementedError
    out = np.zeros((len(xa), len(xb)))
    for i in range(len(xa)):
        for j in range(len(xb)):
            error4elems=[]
            for x, y in zip(xa[i], xb[j]):
                if x==0.0 and y==0.0:
                    error4elems.append(0.0)
                else:
                    error4elems.append(max([
                        abs((x-y)/x) if x!=0.0 else 0.0,
                        abs((x-y)/y) if y!=0.0 else 0.0
                        ]))

            out[i, j]=max(error4elems)
    return out

def dist4groups_matrix(comps4groups:list[list[Composition]], metric:Literal["cityblock", "chebyshev", "max_sym_relative_error"], 
                       ignore_cross_elemental_set:bool = False, group_cross_elemental_set:bool=False):
    """
    arguments:
        - ignore_cross_elemental_set: 
            - If float, add value to the distance when elemental sets are different
        - group_cross_elemental_set:
            - if elemental sets of a group are all the same.
            - should be False if ignore_cross_elemental_set
    """
    if ignore_cross_elemental_set:
        assert not group_cross_elemental_set, NotImplementedError
    out = np.zeros((len(comps4groups),len(comps4groups)))
    for group_i in range(len(comps4groups)):
        for group_j in range(len(comps4groups)):
            vectors0=comps4groups[group_i]
            vectors1=comps4groups[group_j]
            if group_i==group_j:
                out[group_i, group_j]=np.nan
            elif ignore_cross_elemental_set:
                if set(vectors0[0].elements)==set(vectors1[0].elements):
                    out[group_i, group_j] = distance_matrix(vectors0, vectors1, metric=metric).min()
                else:
                    out[group_i, group_j] = np.nan
            else:
                out[group_i, group_j] = distance_matrix(vectors0, vectors1, metric=metric).min()
    return out

def compare_dupl_groups(comps4groups:list[list[Composition]],
                        group_names:list[str],
                        ignore_cross_elemental_set:bool=False,
                        print_output=True, metrices=['cityblock','chebyshev','MSRE']):
    out=dict()
    for metric in metrices:
        out[metric]=dist4groups_matrix(comps4groups, metric=metric,
                                       ignore_cross_elemental_set=ignore_cross_elemental_set)
        out[metric]=pd.DataFrame(out[metric], 
                                        columns=group_names, index=group_names)
        if print_output:
            print(f"{metric} of groups:")
            print(out[metric])

    return out

def compare_dupl_groups_old2new(dataset, dup_group, entry_idx2group_idx, old_none_groupped=True):
    disagreement=False
    out = dict()
    for old_group_index, old_group in dataset.duplicated_comps_group.items():
        if set(old_group.keys())!=set(dup_group[entry_idx2group_idx[old_group_index]]):
            disagreement=True
            print(f"old group[{old_group_index}] indices: {old_group.keys()}")
            assert old_group_index in old_group.keys()
            # old_group.keys() are entry indices of groupped entries.
            new_groups_overlapped = list(set([entry_idx2group_idx[i] for i in old_group.keys()]))
            print(f"new groups overlapped: [{new_groups_overlapped}] indices: {[dup_group[i] for i in new_groups_overlapped]}")
            
            relative_groups_name = [f"old_group{old_group_index}"]+new_groups_overlapped

            relative_groups=[[dataset[idx]['comps_pymatgen'] for idx in old_group.keys()]]
            for new_group_idx in new_groups_overlapped:
                relative_groups=relative_groups+[[dataset[i]['comps_pymatgen'] for i in dup_group[new_group_idx]]]
            
            out[old_group_index] = dict(
                dist_matrix=compare_dupl_groups(relative_groups, relative_groups_name))
    if old_none_groupped:
        for i, row in dataset._df.iterrows():
            if i not in dataset.duplicated_comps: # means not groupped (in new groups, len(group)=1)
                if set([i])!=set(dup_group[entry_idx2group_idx[i]]):
                    disagreement=True
                    print(f"old non-groupped[{i}]: new group idx {entry_idx2group_idx[i]}")
                    print(f"new group idx {entry_idx2group_idx[i]}: {dup_group[entry_idx2group_idx[i]]}")


    if disagreement:
        return out
                    
def distance_matrix(comps0:list[Composition] |list[list[float]],
                    comps1:list[Composition] |list[list[float]],
                    metric:Literal["chebyshev", "cityblock", "euclidean", "L1", "l1", "L2","l2", "L_infty", "maximum", "tchebychev", "l_infty", "max_sym_relative_error", "MSRE"], elemental_set:list[str]|None = None,
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
            - "max_sym_relative_error": $max(max_i|x_i-y_i|/x_i, max_i|x_i-y_i|/y_i))$
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
    alias_map = alias_map | {al: "max_sym_relative_error" for al in ('MSRE', 'msre')}
    metric_out = alias_map.get(metric, metric)
    if metric_out == "max_sym_relative_error":
        return max_relative_error(comps_list[0], comps_list[1], symmetric=True)
    else:
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
        
def make_duplicates_group(index, 
                          dist_matrices:dict, elements_sets_rows:list,
                          dist_cutoffs:dict=dict(chebyshev=0.01, cityblock=0.02, MSRE=0.02),
                            cross_elements_set=False, verbose=True):
    if verbose:
        print(f"start to make duplicates group; dist_criteria: {dist_cutoffs}")
    assert index.tolist()==list(range(len(elements_sets_rows)))
    assert all(m in dist_matrices for m in dist_cutoffs.keys())
    criteria_metrices=dist_cutoffs.keys()
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
            if all(dist_matrices[m][i, j] < dist_cutoffs[m] for m in criteria_metrices):
                if cross_elements_set or elements_sets_rows[i]==elements_sets_rows[j]:
                    if j<=i: # just double check
                        assert i in duplicates_group[group_rows[j]]
                    elif j>i:
                        duplicates_group[current_group_idx].add(j)
                        group_rows[j]=current_group_idx # TODO: maybe I would store multiple values to optimize merge_overlap()
                    else:
                        raise ValueError
                                    
    return merge_overlap(index, duplicates_group, group_rows)