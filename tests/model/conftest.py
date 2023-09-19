import pytest

from madewithml import predict
from madewithml.predict import TorchPredictor


def pytest_addoption(parser):
    parser.addoption("--run-id", action="store", default=None, help="Run ID of model to use.")


@pytest.fixture(scope="module")
def run_id(request):
    return request.config.getoption("--run-id")


@pytest.fixture(scope="module")
def predictor(run_id):
    best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)
    return predictor
