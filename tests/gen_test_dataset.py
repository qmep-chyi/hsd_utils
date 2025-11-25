from pathlib import Path
import importlib.resources as resources

import pandas as pd


def test_dataset_gen(full_dataset_pth, from_pth=None, to_pth=None, keep_comps=True):
    """update test set
    
    test set is a small subset to test repository
    * because full dataset is not public yet..
    
    arguments:
    * from_pth: old, test dataset path
    * to_pth: new test dataset path
    * full_dataset_pth: full dataset (which is private)

    * keep_coms:
        * look value of "composition" column only,
        * find first row that matches "composition"
        * On 25 Nov, composition of test set are;
            ['MoReRuRhPt',
            'MoReRuIrPt',
            'TiZrNbHfTaH10',
            '(ScZrNb)0.55[RhPd]0.45',
            '(ScZrNb)0.60[RhPd]0.40',
            'Ru0.20Rh0.20Pd0.20Ir0.20Pt0.20Sb',
            'Ru0.15Rh0.15Pd0.15Ir0.15Pt0.40Sb',
            'Ru0.075Rh0.075Pd0.075Ir0.075Pt0.70Sb',
            'Ta1/6Nb2/6Hf1/6Zr1/6Ti1/6',
            'Nb0.34Ti0.33Zr0.14Ta0.11Hf0.08',
            'Ag0.15Sn0.16Pb0.15Bi0.14In0.40Te',
            'Ta1/6Nb2/6Hf1/6Zr1/6Ti1/6',
            'Ta1/6Nb2/6Hf1/6Zr1/6Ti1/6',
            'Ta1/6Nb2/6Hf1/6Zr1/6Ti1/6',
            'HfMoNbTiZr',
            'FeZr2',
            'CuZr2',
            'Nb60Re10Zr10Hf10Ti10',
            'Re0.35Os0.35Mo0.10W0.10Zr0.10',
            'Ru0.35Os0.35Mo0.10W0.10Zr0.10',
            'Mo0.11W0.11V0.11Re0.34B0.33',
            '(MoReRu)(1−2x)/3(PdPt)xCy, x = 0.042, y = 0.3',
            '(MoReRu)(1−2x)/3(PdPt)xCy, x = 0.083, y = 0.44']
    """
    # initialize paths
    assert keep_comps, NotImplementedError
    default_testset_path:Path
    with resources.as_file(resources.files("hsdu.data.tests") /"test_dataset.csv") as path:
        default_testset_path = path
    if from_pth is None:
        from_pth=default_testset_path
    else:
        from_pth=Path(from_pth)
        if from_pth.is_absolute():
            pass
        else:
            from_pth=default_testset_path.parent.joinpath(from_pth)
        assert from_pth.is_file()
    if to_pth is None:
        to_pth=default_testset_path
    
    full_dataset_pth=Path(full_dataset_pth)
    assert full_dataset_pth.is_file()
    
    dataset=pd.read_csv(full_dataset_pth)

    old_test=pd.read_csv(from_pth)
    
    test_list=[]
    for idx, row in old_test.iterrows():
        for fidx, frow in dataset.iterrows():
            if frow["composition"]==row["composition"]:
                test_list.append(fidx)
                break
    
    dataset.loc[test_list].to_csv(to_pth, index=False)

if __name__=="__main__":
    test_dataset_gen(r"merged_dataset_forward.csv")
