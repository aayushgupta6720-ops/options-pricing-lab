import pytest

from optlab import calibration


@pytest.fixture(autouse=True, scope="session")
def fewer_rough_paths():
    """Rough Bergomi fits run on 20,000 paths in production; 4,000 is plenty to exercise the code."""
    calibration.ROUGH_PATHS, original = 4_000, calibration.ROUGH_PATHS
    yield
    calibration.ROUGH_PATHS = original
