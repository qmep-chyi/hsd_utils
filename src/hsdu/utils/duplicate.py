#%%
import argparse
from typing import Literal
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element
from hsdu.utils.conversion_utils import element_list_iupac_ordered, OneHotFracCodec

class DuplicatesWarning(UserWarning):
    pass

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

def merge_overlap(index, duplicates_group, group_rows, mode='itself'):
    """ merging overlapped groups
    
    leave 'first group' and remove others.
    """
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
                for idx1 in duplicates_group[j]:
                    # if other group's idx not in first group, append.
                    if idx1 not in duplicates_group[first_group]:
                        duplicates_group[first_group].append(idx1)
                for k in duplicates_group[j]:
                    group_rows[k]=first_group
                duplicates_group.pop(j)

    if overlaps:
        return merge_overlap(index, duplicates_group, group_rows)
    else:
        # if recursion finished, mode=='itself', all the index of the current set belongs to only one group.
        #assert sum(len(duplicates_group[gidx]) for gidx in current_set_groups) == len(index) or mode=='other'
        if mode=='itself':
            current_set_groups = set([group_rows[i] for i in index])
            sum_groupped_indices=0
            for gidx in current_set_groups:
                assert all([i in index for i in duplicates_group[gidx]])
                sum_groupped_indices+=len(duplicates_group[gidx])
            assert sum_groupped_indices==len(index)
                

        for i in index:
            assert sum([1 if i in v else 0 for v in duplicates_group.values()])==1 or mode=='other'
        return duplicates_group, group_rows

def group_duplicates_loop(index0, index1, dupl_groups, idx2gid, dist_cutoffs, dist_matrices, mode):
    """group close enough entries

    Args:
        dupl_groups: mapping, duplicates `group_idx` to 'index0'
        idx2gid: mapping, 'index0' to 'group_idx'
        mode: Literal["itself" | "other"]
            * if mode=='other'
                * 'group index' is just the index of the 'other' datatable.
                * do not merge groups (assuming 'other' has no duplicates internally)
    """
    criteria_metrices=[k for k, v in dist_cutoffs.items() if v is not None]
    if len(dupl_groups)!=0:
        # note that keys of dupl_groups are not 
        last_dup_group_idx = max(dupl_groups.keys())
    else:
        # first time, running with empty dupl_groups and idx2gid
        assert len(dupl_groups)==0 and len(idx2gid)==0
        last_dup_group_idx = -1
    current_group_idx = last_dup_group_idx+1
    
    assert all([(len(index0), len(index1))==matrix.shape for _, matrix in dist_matrices.items()])

    for i in range(len(index0)):
        # init group
        if idx2gid.get(index0[i]) is None: 
            # init a new group
            last_dup_group_idx+=1 # is a new group_idx
            assert dupl_groups.get(last_dup_group_idx) is None or mode=='other'
            if mode=='itself':
                dupl_groups.setdefault(last_dup_group_idx, [index0[i]])
                current_group_idx = last_dup_group_idx
                idx2gid.setdefault(index0[i], last_dup_group_idx)
            elif mode=='other':
                # in this case, duplicates_group maps index1(key) to duplicate entries in index0(value, as a list[int])
                current_group_idx=None
                idx2gid[index0[i]]=None
            else:
                raise ValueError(mode)
        else:
            current_group_idx = idx2gid[index0[i]]
        
        # process duplicates
        for j in range(len(index1)):
            if all(dist_matrices[m][i, j] < dist_cutoffs[m] for m in criteria_metrices):
                if mode=='other':
                    dupl_groups[index1[j]]=dupl_groups.get(index1[j], [])
                    dupl_groups[index1[j]].append(index0[i])
                    assert idx2gid.get(index1[j]) is None
                    idx2gid[index0[i]]=index1[j] # TODO: maybe I would store multiple values to optimize merge_overlap()
                elif mode=='itself':
                    if j<=i: # just double check
                        assert index0[i] in dupl_groups[idx2gid[index1[j]]]
                    elif j>i:
                        if index1[j] not in dupl_groups[current_group_idx]:
                            dupl_groups[current_group_idx].append(index1[j])
                        idx2gid[index1[j]]=current_group_idx # TODO: maybe I would store multiple values to optimize merge_overlap()
                    else:
                        raise ValueError
                else:
                    raise ValueError
    return dupl_groups, idx2gid

def make_duplicates_group(index0:list[int], index1:list[int],
                          dupl_groups, idx2group_idx,
                          dist_matrices:dict,
                          dist_cutoffs:dict=dict(chebyshev=0.01, cityblock=0.02, MSRE=0.02),
                          verbose=True,
                          mode:Literal['itself', 'other']='itself'):
    """
    Args:
    - index: if tuple(list, list), compare two different dataset. else, internally.
    
    Return:
    - merge_overlap(): recursively merge groups if close overlapped. finally returns---
        - duplicates_group: dictionary, 
            - key: group_idx
            - value: index of entries belongs to the group
        - group_rows: list of len(dataset) elements, the duplicates group idx of dataset[idx] belongs to.
    """
    if verbose:
        print(f"start to make duplicates group; dist_criteria: {dist_cutoffs}")

    assert all(m in dist_matrices for m in dist_cutoffs.keys())

    duplicates_group, group_rows = group_duplicates_loop(index0, index1, dupl_groups, idx2group_idx, dist_cutoffs, dist_matrices, mode)
    
    if mode=='other':
        if len(duplicates_group)!=len(index0):
            warnings.warn('multiple self entries are grouped as duplicates of other entries. If duplicates of self and other datatable are merged in same way, it shoud not happens', DuplicatesWarning)
        return duplicates_group, group_rows
    elif mode=='itself':
        return merge_overlap(index0, duplicates_group, group_rows, mode)