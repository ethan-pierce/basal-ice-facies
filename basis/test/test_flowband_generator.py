import pytest

from src.flowband import Flowband, FlowbandGenerator

def test_always_passes():
    '''This test always passes.'''
    assert True

@pytest.fixture
def flowband():
    return Flowband(None)

class TestFlowband:
    '''Test the Flowband dataclass.'''

    def test_initialization(self, flowband):
        '''Test initialization on an artificial grid.'''
        assert flowband.grid == None


test_vars = ['ice_thickness', 'velocity_x', 'velocity_y']
test_files = ['./test/test_data/' + i + '.tif' for i in test_vars]

@pytest.fixture
def generator():
    return FlowbandGenerator(test_files)

class TestFlowbandGenerator:
    '''Test the FlowbandGenerator utility.'''

    def test_initialization(self, generator):
        '''Test initialization with given input files.'''

        for var in test_vars:
            '''Test that variable list is set up.'''
            assert var in generator.variables

        for var in test_vars:
            '''Test that fields exist and are not None.'''
            assert type(getattr(generator, var)) is not None

        for var in test_vars:
            '''Test that input fields have the same shape as the first variable listed.'''
            assert getattr(generator, var).shape == getattr(generator, test_vars[0]).shape

    
