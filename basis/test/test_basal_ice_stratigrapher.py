"""Test the BasalIceStratigrapher class using a synthetic model configuration."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from landlab import RasterModelGrid

from src.basal_ice_stratigrapher import BasalIceStratigrapher

def test_always_passes():
    """This test always passes."""
    assert True

config = 'test/test_data/input_file.toml'

def test_initialize():
    """Test model initialization routines."""
    basis = BasalIceStratigrapher(config)

    assert basis.grid.number_of_nodes == 16
    assert basis.parameters['erosion_coefficient'] == 4e-4
    assert 'soil__depth' in basis.grid.at_node.keys()

def test_calc_erosion_rate():
    """Test the erosion rate calculation."""
    basis = BasalIceStratigrapher(config)
    basis.calc_erosion_rate()

    assert_array_equal(basis.grid.at_node['erosion__rate'], [4e-2, 4e-2, 4e-2, 4e-2,
                                                             4e-2, 4e-2, 4e-2, 4e-2,
                                                             4e-2, 4e-2, 4e-2, 4e-2,
                                                             4e-2, 4e-2, 4e-2, 4e-2])

def test_calc_melt_rate():
    """Test the melt rate calculation."""
    basis = BasalIceStratigrapher(config)
    basis.calc_melt_rate()

    assert_array_almost_equal(basis.grid.at_node['subglacial_melt__rate'], [0.01959, 0.01959, 0.01959, 0.01959,
                                                                            0.01959, 0.01959, 0.01959, 0.01959,
                                                                            0.01959, 0.01959, 0.01959, 0.01959,
                                                                            0.01959, 0.01959, 0.01959, 0.01959])
