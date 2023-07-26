import great_expectations as ge
import pandas as pd
import pytest


def pytest_addoption(parser):
    """Add option to specify dataset location when executing tests from CLI.
    Ex: pytest --dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings
    """
    parser.addoption("--dataset-loc", action="store", default=None, help="Dataset location.")


@pytest.fixture(scope="module")
def df(request):
    dataset_loc = request.config.getoption("--dataset-loc")
    df = ge.dataset.PandasDataset(pd.read_csv(dataset_loc))
    return df
