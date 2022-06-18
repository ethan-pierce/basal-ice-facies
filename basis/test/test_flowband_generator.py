import pytest

from src.flowband import FlowbandGenerator

def test_always_passes():
    '''This test always passes.'''
    assert True

test_config = 'test/test_data/example_config.yml'

@pytest.fixture
def generator():
    return FlowbandGenerator(test_config)

class TestFlowbandGenerator:
    '''Test the FlowbandGenerator utility.'''

    def test_config_file_io(self, generator):
        '''Test initialization with example configuration files.'''
        assert generator

        for var in ['ice_thickness', 'velocity_x', 'velocity_y']:
            assert var in generator.variables

        assert generator.template == 'ice_thickness'
        assert len(generator.shape) == 2
        assert len(generator.lower_left) == 2
        assert len(generator.resolution) == 2
        assert generator.grid.shape == generator.shape

        for var in generator.variables:
            assert var in generator.grid.at_node.keys()

    def test_starting_locations(self, generator):
        '''Test starting locations are initialized correctly.'''

        assert len(generator.initial_nodes) > 0

        for node in generator.initial_nodes:
            assert generator.grid.at_node[generator.init_field][node] > generator.init_min
            assert generator.grid.at_node[generator.init_field][node] < generator.init_max
