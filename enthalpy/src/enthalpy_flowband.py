import numpy as np
from landlab import RasterModelGrid
from typing import Callable

class EnthalpyFlowbandState:
    '''Stores all state variables and calculates diagnostic quantities.

    The units for the model are meters/kilograms/seconds.
    All temperatures are in Kelvin, and all constants are taken for ice at
    273 K, as per chapter 9 of Cuffey and Paterson (2010). The enthalpy-to-
    temperature conversions rely on a reference temperature of 273 K, with a
    corresponding reference pressure of 612 Pa (Engineering Toolbox, 2017).

    Attributes:
        rho_ice: density of ice
        heat_capacity: specific heat capacity of ice at 273 K
        latent_heat: latent heat of fusion for ice at 273 K
        conductivity: thermal conductivity of ice at 273 K
        rho_water: density of water
        gravity: gravitational acceleration
        clapeyron: slope of the Clausius-Clapeyron relationship
        temperature_ref: reference temperature for Clapeyron equation
        pressure_ref: reference pressure for Clapeyron equation
        glens_coeff: ice fluidity coefficient in Glen's flow law
        glens_n: exponent in Glen's flow law
    '''

    rho_ice = 917 # kg m^-3
    heat_capacity = 2097 # J kg^-1 K^-1
    latent_heat = 3.34e5 # J kg^-1
    conductivity = 2.10 # W m^-1 K^-1
    rho_water = 1000 # kg m^-3
    gravity = 9.8 # m^2 s^-1
    clapeyron = 7.9e-8 # K Pa^-1
    temperature_ref = 273 # K
    pressure_ref = 612 # Pa
    glens_coeff = 2.4e-24 # Pa^-3 s^-1
    glens_n = 3

    def __init__(self, nx: int, nz: int, dx: float, dz: float, t_final: float, dt: float):
        '''Initializes the model domain with dimensions and grid resolution.'''

        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.t_final = t_final
        self.dt = dt
        self.grid = RasterModelGrid((nx, nz), (dx, dz))
        self.time_array = np.arange(1, t_final + dt, dt)

        empty = np.empty(self.grid.number_of_nodes)

        # Call self.set_initial_surface() to populate these fields
        self.surface = empty
        self.in_glacier = empty

        # Call self.set_pressure_melting_point() to populate these fields
        self.hydrostatic_pressure = empty
        self.melt_point = empty

        # Call self.set_initial_velocity() to populate these fields
        self.velocity_x = empty
        self.velocity_z = empty

        # Call self.calculate_internal_energy() to populate these fields
        self.strain_rate = np.empty((4, self.grid.number_of_nodes))
        self.deviatoric_stress = np.empty((4, self.grid.number_of_nodes))
        self.internal_energy = empty

        # Call self.set_air_temperature() to populate this field
        self.air_temperature = np.empty_like(self.time_array)

        # Simulation variables
        self.is_temperate = empty
        self.enthalpy = empty
        self.temperature = empty
        self.porosity = empty
        self.effective_pressure = empty

    def set_initial_surface(self, surface_function: Callable[np.ndarray, np.ndarray]):
        '''Set surface height and create a boolean mask of the glacier.'''

        self.surface = surface_function(self.grid.node_x)

        for node in np.ravel(self.grid.nodes):
            if self.grid.node_y[node] < self.surface[node]:
                self.in_glacier[node] = 1
            else:
                self.in_glacier[node] = 0

        # Close nodes outside the glacier
        self.grid.set_nodata_nodes_to_closed(self.in_glacier, 0)

    def set_pressure_melting_point(self):
        '''Compute the pressure-melting-point of ice throughout the domain.'''

        self.hydrostatic_pressure = self.rho_ice * self.gravity * (self.surface - self.grid.node_y)
        self.melt_point = (self.temperature_ref - self.clapeyron *
            (self.hydrostatic_pressure - self.pressure_ref))

    def set_initial_velocity(self, velocity_x_function:
                             Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]):
        '''Set the velocity vector field given the coordinates of each node.'''

        self.velocity_x = velocity_x_function(self.grid.node_x, self.grid.node_y, self.surface)
        diff_at_links = self.grid.calc_diff_at_link(self.velocity_x)
        gradient_velocity_x = self.grid.map_mean_of_horizontal_active_links_to_node(diff_at_links)

        # If we assume plug flow and incompressibility, then u_z(z) = -du_x/dx * z
        # This is a sensible default value, but specific instances should override this as needed.
        self.velocity_z = -gradient_velocity_x * self.grid.node_y

    def set_air_temperature(self, temperature_function: Callable[np.ndarray, np.ndarray]):
        '''Pre-compute and store the air temperature above the glacier surface.'''

        self.air_temperature = temperature_function(self.time_array)

    def calc_internal_energy(self):
        '''Compute the strain rate tensor and the deviatoric stress tensor from velocities.'''

        # Notation: u is the x-component of velocity, w is the z-component of velocity
        du_dx = self.grid.map_mean_of_horizontal_active_links_to_node(
                    self.grid.calc_diff_at_link(self.velocity_x)
        )

        du_dz = self.grid.map_mean_of_vertical_active_links_to_node(
                    self.grid.calc_diff_at_link(self.velocity_x)
        )

        dw_dx = self.grid.map_mean_of_horizontal_active_links_to_node(
                    self.grid.calc_diff_at_link(self.velocity_z)
        )

        dw_dz = self.grid.map_mean_of_vertical_active_links_to_node(
                    self.grid.calc_diff_at_link(self.velocity_z)
        )

        # Convention: store tensor components in C-style (row-major) ordering, such that
        # index 0 is xx or du/dx, 1 is xz or du/dz, 2 is zx or dw/dx, and 3 is zz or dw/dz
        self.strain_rate[0] = du_dx
        self.strain_rate[1] = 0.5 * (du_dz + dw_dx)
        self.strain_rate[2] = self.strain_rate[1].copy()
        self.strain_rate[3] = dw_dz

        effective_strain = np.sqrt(0.5 * (self.strain_rate[0]**2 + self.strain_rate[1]**2
                                   + self.strain_rate[2]**2 + self.strain_rate[3]**2))

        for i in range(4):
            self.deviatoric_stress[i] = (self.glens_coeff**(-1 / self.glens_n) *
                                         effective_strain**(1 / (self.glens_n - 1)) *
                                         self.strain_rate[i])

        self.internal_energy = np.sum([self.strain_rate[i] * self.deviatoric_stress[i]
                                       for i in range(4)], axis = 0)

    def partition_domain(self):
        '''Partition the glacier domain into temperate and cold regions.'''

        threshold = self.rho_ice * self.heat_capacity * (self.melt_point - self.temperature_ref)
        self.is_temperate = np.where(self.enthalpy >= threshold, 1, 0)

    def calc_temperature(self):
        '''Calculate the temperature field for the glacier, given enthalpy.'''

        self.temperature = np.where(self.is_temperate, self.melt_point,
                                    self.temperature_ref + self.enthalpy /
                                    (self.rho_ice * self.heat_capacity))

    def calc_porosity(self):
        '''Calculate the porosity field for the glacier, given enthalpy.'''

        porosity_if_temperate = ((self.enthalpy - (self.rho_ice * self.heat_capacity *
                                 (self.melt_point - self.temperature_ref))) /
                                 (self.rho_water * self.latent_heat))

        self.porosity = np.where(self.is_temperate, porosity_if_temperate, 0)

    def calc_enthalpy(self):
        '''Calculate the enthalpy field for the glacier, given temperature and porosity'''

        self.enthalpy = (self.rho_ice * self.heat_capacity * (self.temperature - self.temperature_ref)
                         + self.rho_water * self.latent_heat * self.porosity)

    def set_initial_conditions(self, initial_temperature: float = 268):
        '''Initialize the glacier with a constant temperature (below 273 K) and zero porosity.'''

        self.temperature = np.where(self.in_glacier, initial_temperature, 0)
        self.porosity = np.zeros_like(self.temperature)
        self.calc_enthalpy()
        self.partition_domain()

class EnthalpyFlowbandModel:

    def __init__(self):
        pass

    def assemble_enthalpy_matrix(self):
        pass

    def apply_enthalpy_boundary_conditions(self):
        pass

    def solve_enthalpy_system(self):
        pass

    def advance_enthalpy(self):
        pass

    def assemble_pressure_matrix(self):
        pass

    def apply_pressure_boundary_conditions(self):
        pass

    def solve_pressure_system(self):
        pass

    def advance_pressure(self):
        pass

    def advance_drainage_system(self):
        pass

    def run_until(self, t_final: float):
        pass
