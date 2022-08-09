"""Tracks the evolution of basal sediment entrainment beneath a glacier.

Coupled model of erosion and sediment entrainment beneath an ice sheet or glacier, using the Landlab
framework. The model estimates erosion rates using a power law relationship for bulk erosion
(Herman et al., 2021). Sediment is entrained as part of a frozen fringe (Meyer et al., 2019) at the
ice-till interface, and is then transported within the ice column by vertical regelation
(Iverson and Semmens, 1995). Basal melt removes material at the slip interface, which is assumed to
sit below the frozen fringe layer.

Required input fields:
    glacier__thickness: defined at nodes, the thickness of ice
    glacier__sliding_velocity: defined at links, the velocity of ice at the slip interface
    glacier__effective_pressure: defined at nodes, the difference between overburden and water pressure
    bedrock__geothermal_heat_flux: defined at nodes, the heat flux at the ice-bed interface

Example usage:
    None

Attributes:
    None

Methods:
    None
"""
import numpy as np
import toml
from landlab import RasterModelGrid
import rasterio as rio

class BasalIceStratigrapher:
    """Tracks the evolution of basal sediment entrainment beneath a glacier."""

    def __init__(self, input_file):
        """Initializes the model with a Landlab grid object."""
        with open(input_file) as f:
            inputs = toml.loads(f.read())

        self.grid = RasterModelGrid(inputs['grid']['shape'], inputs['grid']['spacing'])
        self.parameters = inputs['parameters']

        for key in inputs['solution_fields'].keys():
            if key not in self.grid.at_node.keys():
                self.grid.add_zeros(key, at = 'node', units = inputs['solution_fields'][key]['units'])
                self.grid.at_node[key][:] = inputs['solution_fields'][key]['initial']

        for key in inputs['input_fields'].keys():
            if key not in self.grid.at_node.keys():
                with rio.open(inputs['input_fields'][key], 'r') as f:
                    data = f.read(1)

                    if data.shape == self.grid.shape:
                        self.grid.add_field(key, data, at = 'node', units = inputs['solution_fields'][key]['units'])

                    else:
                        raise ValueError("Shape of " + str(key) + " data does not match grid shape.")

    def calc_erosion_rate(self):
        """Calculates the erosion rate as a function of sliding velocity."""
        required = ['soil__depth', 'glacier__sliding_velocity']

        for field in required:
            if field not in self.grid.at_node.keys():
                raise ValueError("Missing " + str(field) + " at nodes.")

        if 'erosion__rate' not in self.grid.at_node.keys():
            self.grid.add_zeros('erosion__rate', at = 'node')

        Ks = self.parameters['erosion_coefficient']
        m = self.parameters['erosion_exponent']
        ub = self.grid.at_node['glacier__sliding_velocity'][:]

        self.grid.at_node['erosion__rate'][:] = (
            Ks * np.abs(ub)**m
        )

    def calc_melt_rate(self):
        """Calculates the basal melt rate as a function of shear stress and heat fluxes."""
        required = ['glacier__sliding_velocity', 'glacier__effective_pressure']

        for field in required:
            if field not in self.grid.at_node.keys():
                raise ValueError("Missing " + str(field) + " at nodes.")

        if 'subglacial_melt__rate' not in self.grid.at_node.keys():
            self.grid.add_zeros('subglacial_melt__rate', at = 'node')

        rho = self.parameters['ice_density']
        L = self.parameters['ice_latent_heat']
        ub = self.grid.at_node['glacier__sliding_velocity'][:]
        N = self.grid.at_node['glacier__effective_pressure'][:]
        mu = self.parameters['friction_coefficient']
        frictional = ub * mu * N
        geothermal = self.parameters['geothermal_heat_flux']

        self.grid.at_node['subglacial_melt__rate'][:] = (frictional + geothermal) / (rho * L)

    def calc_fringe_growth_rate(self):
        """Calculates the growth rate of frozen fringe as a function of melt and pressure."""
        pass

    def calc_regelation_rate(self):
        """Calculates the vertical regelation rate as a function of heat fluxes and pressure."""
        pass

    def calc_advective_deformation(self):
        """Calculates thickening or thinning of ice as a function of the velocity field."""
        pass

    def run_one_step(self, dt):
        """Advances the model forward one time step."""
        pass
