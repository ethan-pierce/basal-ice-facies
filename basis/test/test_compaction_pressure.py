import pytest
import numpy as np
np.random.seed(31400000)

from src.flowband import FlowbandGenerator
from src.enthalpy import CompactionPressureModel

def test_always_passes():
    '''This test always passes.'''
    assert True

test_config = 'test/test_data/example_config.yml'
generator = FlowbandGenerator(test_config)
flowband = generator.construct_flowband('ice_thickness', dt = 10, max_iter = 1000, nz = 100)

@pytest.fixture
def cpm():
    return CompactionPressureModel(flowband)

class TestCompactionPressureModel:
    '''Class to test enthalpy model with compaction-pressure closure.'''

    def test_initialization(self, cpm):
        '''Test initialization.'''
        assert cpm.mesh.shape == (100, 160)
