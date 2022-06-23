import pytest
import numpy as np

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

    def test_break_nodes(self, generator):
        '''Test break condition established correctly.'''

        assert len(generator.break_nodes) > 0

        for node in generator.break_nodes:
            assert generator.grid.at_node['ice_thickness'][node] <= 0

    def test_generate_flowline(self, generator):
        '''Test flowline generation algorithm.'''

        flowline = generator.generate_flowline(100, 1000)

        assert len(flowline.node_id) == len(flowline.node_x)
        assert len(flowline.node_id) == len(flowline.node_y)
        assert len(flowline.node_id) == len(flowline.distance)

        for key in flowline.fields.keys():
            assert len(flowline.fields[key]) == len(flowline.node_id)

        assert flowline.distance[0] == 0.0
        for idx in range(len(flowline.distance)):
            if idx != 0:
                assert(flowline.distance[idx] >= flowline.distance[idx - 1])

    def test_construct_flowband(self, generator):
        '''Test flowband construction and x-z mesh generation.'''
        np.random.seed(31400000)

        mesh = generator.construct_flowband('ice_thickness', dt = 10, max_iter = 1000, nz = 100)

        assert mesh.shape == (100, 160)
        assert np.max(mesh.node_y) > np.max(mesh.at_node['ice_thickness'])

        column = np.where(mesh.node_x == mesh.node_x[50])
        assert np.all(mesh.at_node['ice_thickness'][column] == mesh.at_node['ice_thickness'][column[0]])

        inside = np.where(mesh.at_node['in_domain'] == 1)[0]
        outside = np.where(mesh.at_node['in_domain'] == 0)[0]
        assert len(inside) > 0
        assert len(outside) > 0
        assert len(inside) + len(outside) == mesh.number_of_nodes
