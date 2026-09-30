import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraud.data import generate  # noqa: E402
from fraud.train import train  # noqa: E402


@pytest.fixture(scope="session")
def data():
    return generate(n_customers=300, days=20, seed=1)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, data):
    out = tmp_path_factory.mktemp("artifacts")
    metrics = train(data, out=out)
    return out, metrics
