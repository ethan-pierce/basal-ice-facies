import pytest

from src.flowband import Flowband
from src.enthalpy import CompactionPressureModel

def test_always_passes():
    '''This test always passes.'''
    assert True

test_grid = None

@pytest.fixture
def cpm():
    fb = Flowband(test_grid)
    return CompactionPressureModel(fb)

class TestCompactionPressureModel:
    '''Class to test enthalpy model with compaction-pressure closure.'''

    def test_initialization(self, cpm):
        '''Test initialization.'''
        assert cpm.flowband.grid == test_grid
