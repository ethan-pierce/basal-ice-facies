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

class BasalIceStratigrapher:
    """Tracks the evolution of basal sediment entrainment beneath a glacier."""

    def __init__(self, grid, erosion_coefficient = 0.4e-4,
                             erosion_exponent = 1,
                             ):
        """Initializes the model with a Landlab grid object."""
        self.grid = grid

        if 'glacier__thickness' not in self.grid.at_node.keys():
            raise ValueError("Missing glacier__thickness field at grid nodes.")

        if 'glacier__sliding_velocity' not in self.grid.at_link.keys():
            raise ValueError("")

        self.erosion_coefficient = erosion_coefficient
        self.erosion_exponent = erosion_exponent

    def calc_erosion_rate(self):
        """Calculates the erosion rate as a function of sliding velocity."""


    def calc_melt_rate(self):
        """Calculates the basal melt rate as a function of shear stress and heat fluxes."""
        pass

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
