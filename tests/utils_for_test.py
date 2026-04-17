import importlib.resources as resources
from importlib.resources.abc import Traversable

class TestSnapshotWarning(UserWarning):
    """
    compare with v0.1.3 (2026-04-15) results, with HE-SC 
    if outdated, 
        * make sure it is working as expected
        * and update hard coded result values (go to references of this class)
    #TODO: let's implement snapshot, import some pacakge like `pytest-snapshot`
    """
    pass

def get_package_dataset(version:str='20260415')->Traversable:
    file_name = f'dataset_{version}.csv'
    resource_traversable = resources.files("hsdu.data") / file_name
    print(f'load test dataset version:{version}')

    return resource_traversable