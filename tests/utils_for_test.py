import importlib.resources as resources

class ConsistentResultsError(ValueError):
    """
    compare with v0.1.3 (2026-04-15) results.
    if outdated, update.
    """
    pass

def get_package_dataset(version:str='20260415'):
    file_name = f'dataset_{version}.csv'
    resource_path = resources.files("hsdu.data") / file_name
    print(f'load test dataset version:{version}')

    return resources.as_file(resource_path)