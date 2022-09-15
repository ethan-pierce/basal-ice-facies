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

def test_calc_regelation_rate():
    """Test the particle regelation rate calculations."""
    basis = BasalIceStratigrapher(config)
    basis.grid.at_node['frozen_fringe__thickness'][:] += 1.0
    basis.grid.at_node['dispersed_layer__thickness'][:] += 1.0

    basis.calc_fringe_growth_rate()
    basis.calc_regelation_rate()

    assert_array_almost_equal(basis.grid.at_node['dispersed_layer__growth_rate'][basis.grid.core_nodes],
                              [2.208e-10, 2.208e-10, 2.208e-10, 2.208e-10], 12)

def test_calc_advection():
    """Test advection and deformation calculations."""
    basis = BasalIceStratigrapher(config)
    basis.grid.at_node['dispersed_layer__thickness'][:] += 0.5 * (basis.grid.node_x + basis.grid.node_y)
    basis.grid.at_node['frozen_fringe__thickness'][:] += 1.0 * (basis.grid.node_x + basis.grid.node_y)

    sliding_x = basis.grid.at_node['glacier__sliding_velocity'] * basis.grid.node_x**(1.25) * 1e-2
    basis.grid.add_field('glacier__sliding_velocity_x', sliding_x, at = 'node')

    sliding_y = basis.grid.at_node['glacier__sliding_velocity'] * basis.grid.node_y**(1.25) * 1e-2
    basis.grid.add_field('glacier__sliding_velocity_y', sliding_y, at = 'node')

    basis.calc_melt_rate()
    basis.calc_fringe_growth_rate()
    basis.calc_regelation_rate()
    basis.calc_advection()

    assert_array_almost_equal(basis.grid.at_node['dispersed_layer__advection'][basis.grid.core_nodes],
                              [5.584e-7, 9.432e-7, 9.432e-7, 1.328e-6], 10)

    assert_array_almost_equal(basis.grid.at_node['frozen_fringe__advection'][basis.grid.core_nodes],
                              [1.117e-6, 1.886e-6, 1.886e-6, 2.656e-6], 9)

def test_calc_dynamic_thinning():
    """Test the dynamic thickening and/or thinning calculations."""
    basis = BasalIceStratigrapher(config)
    basis.grid.at_node['dispersed_layer__thickness'][:] += 0.5 * (basis.grid.node_x + basis.grid.node_y)
    basis.grid.at_node['frozen_fringe__thickness'][:] += 1.0 * (basis.grid.node_x + basis.grid.node_y)

    sliding_x = basis.grid.at_node['glacier__sliding_velocity'] * basis.grid.node_x**(1.25) * 1e-2
    basis.grid.add_field('glacier__sliding_velocity_x', sliding_x, at = 'node')

    sliding_y = basis.grid.at_node['glacier__sliding_velocity'] * basis.grid.node_y**(1.25) * 1e-2
    basis.grid.add_field('glacier__sliding_velocity_y', sliding_y, at = 'node')

    basis.calc_melt_rate()
    basis.calc_fringe_growth_rate()
    basis.calc_regelation_rate()
    basis.calc_dynamic_thinning()

    assert_array_almost_equal(basis.grid.at_node['glacier__velocity_divergence'][basis.grid.core_nodes] *
                              basis.grid.at_node['frozen_fringe__thickness'][basis.grid.core_nodes],
                              [2.656e-6, 4.461e-6, 4.461e-6, 6.585e-6], 9)

    assert_array_almost_equal(basis.grid.at_node['glacier__velocity_divergence'][basis.grid.core_nodes] *
                              basis.grid.at_node['dispersed_layer__thickness'][basis.grid.core_nodes],
                              [1.328e-6, 2.230e-6, 2.230e-6, 3.292e-6], 9)
