"""Test the BasalIceStratigrapher class using a synthetic model configuration."""

import pytest
import numpy as np
from landlab import RasterModelGrid

from src.basal_ice_stratigrapher import BasalIceStratigrapher

def test_always_passes():
    """This test always passes."""
    assert True

test_config = 'test/test_data/basis_test_config.yml'

@pytest.fixture
def rmg():
    """Creates a RasterModelGrid object for testing purposes."""
    shape = (4, 4)
    spacing = 1000.

    rmg = RasterModelGrid(shape, spacing)

    h = rmg.add_field('glacier__thickness', np.full(shape, 100.), at = 'node')
    u = rmg.add_field('glacier__sliding_velocity', np.full(rmg.number_of_links, 50.), at = 'link')
    N = rmg.add_field('glacier__effective_pressure', np.full(shape, 1e5), at = 'node')
    qb = rmg.add_field('bedrock__geothermal_heat_flux', np.full(shape, 0.6), at = 'node')

    # Tilt the glacier
    h += rmg.node_x * 5

    return rmg

def test_initialize(rmg):
    """Tests model initialization routines."""
    basis = BasalIceStratigrapher(rmg)
