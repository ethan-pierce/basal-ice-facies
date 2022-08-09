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

    assert_array_equal(basis.grid.at_node['erosion__rate'], [1.256e-9, 1.256e-9, 1.256e-9, 1.256e-9,
                                                             1.256e-9, 1.256e-9, 1.256e-9, 1.256e-9,
                                                             1.256e-9, 1.256e-9, 1.256e-9, 1.256e-9,
                                                             1.256e-9, 1.256e-9, 1.256e-9, 1.256e-9])

def test_calc_melt_rate():
    """Test the melt rate calculation."""
    basis = BasalIceStratigrapher(config)
    basis.calc_melt_rate()

    assert_array_almost_equal(basis.grid.at_node['frictional_heat__flux'], [0.188, 0.188, 0.188, 0.188,
                                                                            0.188, 0.188, 0.188, 0.188,
                                                                            0.188, 0.188, 0.188, 0.188,
                                                                            0.188, 0.188, 0.188, 0.188], 3)

    assert_array_almost_equal(basis.grid.at_node['subglacial_melt__rate'], [8.11e-10, 8.11e-10, 8.11e-10, 8.11e-10,
                                                                            8.11e-10, 8.11e-10, 8.11e-10, 8.11e-10,
                                                                            8.11e-10, 8.11e-10, 8.11e-10, 8.11e-10,
                                                                            8.11e-10, 8.11e-10, 8.11e-10, 8.11e-10])

def test_calc_thermal_gradients():
    """Test thermal gradient calculations."""
    basis = BasalIceStratigrapher(config)
    basis.calc_thermal_gradients()

    assert_array_almost_equal(basis.parameters['entry_pressure'], 68e3)
    assert_array_almost_equal(basis.parameters['fringe_base_temperature'], 272.9394, 3)
    assert_array_almost_equal(basis.parameters['fringe_conductivity'], 4.56, 2)
    assert_array_almost_equal(basis.grid.at_node['fringe__thermal_gradient'], [-0.05, -0.05, -0.05, -0.05,
                                                                               -0.05, -0.05, -0.05, -0.05,
                                                                               -0.05, -0.05, -0.05, -0.05,
                                                                               -0.05, -0.05, -0.05, -0.05], 2)

    assert_array_almost_equal(basis.grid.at_node['transition_temperature'], [272.9394, 272.9394, 272.9394, 272.9394,
                                                                             272.9394, 272.9394, 272.9394, 272.9394,
                                                                             272.9394, 272.9394, 272.9394, 272.9394,
                                                                             272.9394, 272.9394, 272.9394, 272.9394], 3)

def test_calc_fringe_growth():
    """Test frozen fringe calculations."""
    basis = BasalIceStratigrapher(config)
    basis.calc_fringe_growth_rate()

    assert_array_almost_equal(basis.grid.at_node['fringe__undercooling'], [1.0008, 1.0008, 1.0008, 1.0008,
                                                                           1.0008, 1.0008, 1.0008, 1.0008,
                                                                           1.0008, 1.0008, 1.0008, 1.0008,
                                                                           1.0008, 1.0008, 1.0008, 1.0008], 4)

    assert_array_almost_equal(basis.grid.at_node['fringe__saturation'], [0.001, 0.001, 0.001, 0.001,
                                                                         0.001, 0.001, 0.001, 0.001,
                                                                         0.001, 0.001, 0.001, 0.001,
                                                                         0.001, 0.001, 0.001, 0.001], 3)

    assert_array_almost_equal(basis.grid.at_node['nominal__heave_rate'], [1.655e-9, 1.655e-9, 1.655e-9, 1.655e-9,
                                                                          1.655e-9, 1.655e-9, 1.655e-9, 1.655e-9,
                                                                          1.655e-9, 1.655e-9, 1.655e-9, 1.655e-9,
                                                                          1.655e-9, 1.655e-9, 1.655e-9, 1.655e-9])

    assert_array_almost_equal(basis.grid.at_node['flow__resistance'], [0.07, 0.07, 0.07, 0.07,
                                                                       0.07, 0.07, 0.07, 0.07,
                                                                       0.07, 0.07, 0.07, 0.07,
                                                                       0.07, 0.07, 0.07, 0.07], 3)

    assert_array_almost_equal(basis.grid.at_node['fringe__heave_rate'], [-1.095e-8, -1.095e-8, -1.095e-8, -1.095e-8,
                                                                         -1.095e-8, -1.095e-8, -1.095e-8, -1.095e-8,
                                                                         -1.095e-8, -1.095e-8, -1.095e-8, -1.095e-8,
                                                                         -1.095e-8, -1.095e-8, -1.095e-8, -1.095e-8])

    assert_array_almost_equal(basis.grid.at_node['fringe__growth_rate'], [2.173e-5, 2.173e-5, 2.173e-5, 2.173e-5,
                                                                          2.173e-5, 2.173e-5, 2.173e-5, 2.173e-5,
                                                                          2.173e-5, 2.173e-5, 2.173e-5, 2.173e-5,
                                                                          2.173e-5, 2.173e-5, 2.173e-5, 2.173e-5])

def test_advective_deformation():
    """Test advective deformation calculation."""
    basis = BasalIceStratigrapher(config)
    basis.calc_advective_deformation()

    pass
