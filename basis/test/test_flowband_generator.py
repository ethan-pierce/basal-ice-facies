import pytest

from src.flowband import Flowband

def test_always_passes():
    assert True

@pytest.fixture
def flowband():
    return Flowband(None)

class TestFlowband:
    def test_initialization(self, flowband):
        assert flowband.grid == None
