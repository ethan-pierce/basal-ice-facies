import numpy as np
from landlab import RasterModelGrid
from typing import Callable
from dataclasses import dataclass

@dataclass
class BoundaryCondition:
    '''Class for holding boundary conditions for a specified field variable.'''
    field: str
    inflow_boundary: np.ndarray
    outflow_boundary: np.ndarray
    top_boundary: np.ndarray
    bottom_boundary: np.ndarray

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
        drainage_exponent: exponent on porosity in the compaction-pressure model (Hewitt and Schoof, 2017)
        drainage_coeff: coefficient for the compaction-pressure model (Hewitt and Schoof, 2017)
        water_viscosity: viscosity of water
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
    drainage_exponent = 2
    drainage_coeff = 1e-12 #m^2
    water_viscosity = 1.8e-3 # Pa s

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
        self.effective_viscosity = empty
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
        self.water_flux = empty

    def set_initial_surface(self, surface_function: Callable[np.ndarray, np.ndarray]):
        '''Set surface height and create a boolean mask of the glacier.'''

        self.surface = surface_function(self.grid.node_x)

        for node in np.ravel(self.grid.nodes):
            if self.grid.node_y[node] < self.surface[node]:
                self.in_glacier[node] = 1
            else:
                self.in_glacier[node] = 0

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

        self.effective_viscosity = (self.glens_coeff**(-1 / self.glens_n) *
                                    effective_strain**(1 / (self.glens_n - 1)))

        for i in range(4):
            self.deviatoric_stress[i] = self.effective_viscosity * self.strain_rate[i]

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

        # Ensure that the temperature stays below the pressure-melting-point
        self.temperature = np.where(self.temperature > self.melt_point, self.melt_point, self.temperature)

    def calc_porosity(self):
        '''Calculate the porosity field for the glacier, given enthalpy.'''

        porosity_if_temperate = ((self.enthalpy - (self.rho_ice * self.heat_capacity *
                                 (self.melt_point - self.temperature_ref))) /
                                 (self.rho_water * self.latent_heat))

        self.porosity = np.where(self.is_temperate, porosity_if_temperate, 0)

    def calc_enthalpy(self):
        '''Calculate the enthalpy field for the glacier, given temperature and porosity.'''

        self.enthalpy = (self.rho_ice * self.heat_capacity * (self.temperature - self.temperature_ref)
                         + self.rho_water * self.latent_heat * self.porosity)

    def set_initial_conditions(self, initial_temperature: float = 268):
        '''Initialize the glacier with a constant temperature (below 273 K) and zero porosity.'''

        self.temperature = np.where(self.in_glacier, initial_temperature, 0)
        self.porosity = np.zeros_like(self.temperature)
        self.water_flux = np.zeros_like(self.temperature)
        self.effective_pressure = self.hydrostatic_pressure.copy()
        self.calc_enthalpy()
        self.partition_domain()

class EnthalpyFlowbandExplicit:
    '''Models the time evolution of enthalpy along a glacier flowband.

    Uses a compaction-pressure drainage model as a closure term in the temperate part of the domain.
    See Hewitt and Schoof (2017) for a detailed description of the algorithm. Uses central finite
    differences in space and a forward Euler discretization in time, subject to a CFL condition.

    Attributes:
        None
    '''

    def __init__(self, state: EnthalpyFlowbandState, CFL: float,
                       temperature_bc: BoundaryCondition,
                       porosity_bc: BoundaryCondition):
        '''Initialize the enthalpy model with a state handler.'''

        self.state = state
        self.time_array = self.state.time_array
        self.time_step = 0.0
        self.time_elapsed = 0.0
        self.past_time_steps = []
        self.CFL = CFL
        empty = np.empty(self.state.grid.number_of_nodes)

        # Arrays for boundary conditions
        self.top_boundary = None
        self.bottom_boundary = None
        self.outflow_boundary = None
        self.inflow_boundary = None

        self.top_bc = None
        self.bottom_bc = None
        self.outflow_bc = None
        self.inflow_bc = None

        # Boundary conditions
        self.temperature_bc = temperature_bc
        self.porosity_bc = porosity_bc

        # Components of the enthalpy equation
        self.advection = empty
        self.diffusion = empty
        self.water_flux = empty
        self.enthalpy_slope = empty

    def knit(self, node: int) -> tuple[float, float]:
        '''Identify the row and column of a node, given the node ID.'''

        coords = np.column_stack(np.where(self.state.grid.nodes == node))[0]
        return coords

    def calc_dx(self, field: str) -> np.ndarray:
        '''Estimate the x-derivative of a given field.'''

        if isinstance(field, str) == True:
            try:
                field_array = getattr(self.state, field)
            except:
                raise AttributeError('EnthalpyFlowbandState has no field named ' + str(field))
        else:
            field_array = field

        field_diff = self.state.grid.calc_grad_at_link(field_array)
        field_dx = self.state.grid.map_mean_of_horizontal_links_to_node(field_diff)

        return field_dx

    def calc_dz(self, field) -> np.ndarray:
        '''Estimate the y-derivative of a given field.'''

        if isinstance(field, str) == True:
            try:
                field_array = getattr(self.state, field)
            except:
                raise AttributeError('EnthalpyFlowbandState has no field named ' + str(field))
        else:
            field_array = field

        field_dz = np.gradient(np.reshape(field, self.state))

        return field_dz

    def calc_local_enthalpy(self, temperature, porosity):
        '''Calculate enthalpy at a point, given temperature and porosity.'''

        temperature_component = (self.state.rho_ice * self.state.heat_capacity *
                                 (temperature - self.state.temperature_ref))
        porosity_component = self.state.rho_water * self.state.latent_heat * porosity
        enthalpy = temperature_component + porosity_component
        return enthalpy

    def identify_boundaries(self, surface_function: Callable[np.ndarray, np.ndarray]):
        '''Identify node ID's for nodes nearest to each boundary.'''

        x_coords = self.state.grid.node_x[0:self.state.nx]
        z_coords = np.arange(0, self.state.nz + self.state.dz, self.state.dz)
        surface = surface_function(x_coords)

        self.top_boundary = self.state.grid.find_nearest_node((x_coords, surface), mode = 'clip')
        self.bottom_boundary = self.state.grid.find_nearest_node((x_coords, 0), mode = 'clip')
        self.inflow_boundary = self.state.grid.find_nearest_node((0, z_coords), mode = 'clip')
        self.outflow_boundary = self.state.grid.find_nearest_node((np.max(x_coords), z_coords), mode = 'clip')

    def compute_advection(self):
        '''Compute the advective terms in the enthalpy equation.'''

        enthalpy_dx = self.calc_dx('enthalpy')
        enthalpy_dz = self.calc_dz('enthalpy')
        advection_x = self.state.velocity_x * enthalpy_dx
        advection_y = self.state.velocity_z * enthalpy_dz
        self.advection = advection_x + advection_y

    def compute_diffusion(self):
        '''Compute the diffusive term in the enthalpy equation.'''

        temperature_dz = self.calc_dz('temperature')
        diffusive_flux = self.state.conductivity * self.calc_dz(temperature_dz)
        self.diffusion = diffusive_flux

    def update_compaction_pressure(self):
        '''Run the inner compaction-pressure model to solve for water flux and effective pressure.'''

        baseline_pressure = self.state.effective_pressure.copy()

        # Enforce effective pressure = 0 at the surface
        for node in self.top_boundary:
            baseline_pressure[node] = 0.0

        pressure_gradient = self.calc_dz(baseline_pressure)

        # Enforce d(effective pressure) / dz = 0 at the base
        for node in self.bottom_boundary:
            pressure_gradient[node] = 0.0

        inner_function = ((self.state.rho_water - self.state.rho_ice) * self.state.gravity +
                          pressure_gradient)
        laplacian = self.calc_dz(inner_function)
        diffusivity = self.state.effective_viscosity * self.state.drainage_coeff / self.state.water_viscosity

        compaction_pressure = diffusivity * laplacian

        with np.errstate(divide = 'ignore'):
            self.state.effective_pressure = np.where(self.state.porosity > 0,
                                                     compaction_pressure / self.state.porosity,
                                                     self.state.hydrostatic_pressure)

    def compute_water_flux(self):
        '''Compute the drainage term in the enthalpy equation.'''

        flux_coeff = self.state.rho_water * self.state.latent_heat
        water_flux_coeff = ((self.state.drainage_coeff *
                             self.state.porosity**self.state.drainage_exponent) /
                             self.state.water_viscosity)
        pressure_dz = self.calc_dz('effective_pressure')
        water_flux_array = water_flux_coeff * ((self.state.rho_water - self.state.rho_ice) *
                                                self.state.gravity + pressure_dz)
        self.water_flux = water_flux_array

    def enforce_enthalpy_BCs(self):
        '''Calculate Dirichlet boundary conditions on enthalpy from temperature and porosity.'''

        for idx in range(len(self.top_boundary)):
            self.state.enthalpy[self.top_boundary[idx]] = self.top_bc[idx]

        for idx in range(len(self.bottom_boundary)):
            self.state.enthalpy[self.bottom_boundary[idx]] = self.bottom_bc[idx]

        for idx in range(len(self.inflow_boundary)):
            self.state.enthalpy[self.inflow_boundary[idx]] = self.inflow_bc[idx]

        for idx in range(len(self.outflow_boundary)):
            self.state.enthalpy[self.outflow_boundary[idx]] = self.outflow_bc[idx]

    def update_enthalpy_slope(self):
        '''Update the first time derivative of the enthalpy field.'''

        self.enforce_enthalpy_BCs()
        self.compute_advection()
        self.compute_diffusion()
        self.update_compaction_pressure()
        self.compute_water_flux()

        self.enthalpy_slope = (self.state.internal_energy + self.diffusion -
                               self.advection - self.water_flux)

    def set_time_step(self):
        '''Set the appropriate time step based on a modified CFL condition.'''

        self.time_step = (self.CFL * self.state.dx * self.state.dz) / np.max(self.enthalpy_slope)
        self.past_time_steps.append(self.time_step)

    def update_state(self):
        '''Update the state handler based on the enthalpy field calculated for the next time step.'''

        self.set_time_step()
        self.state.enthalpy = self.enthalpy_slope * self.time_step
        self.time_elapsed += self.time_step

    def run_one_step(self, stop_time_index: int):
        '''Advance the model state by one time step.'''

        stop_at_time = self.time_array[stop_time_index]

        while self.time_elapsed < stop_at_time:
            self.update_enthalpy_slope()
            self.update_state()
            self.state.partition_domain()
            self.state.calc_porosity()
            self.state.calc_temperature()

    def run_all_steps(self, verbose_at_interval = False):
        '''Advance the model through all of the time steps in state.time_array.'''

        for idx in range(len(self.time_array)):
            self.run_one_step(idx)

            if verbose_at_interval:
                if idx % verbose_at_interval == 0:
                    print('Time elapsed: ' + str(np.round(self.time_elapsed / 86400, 2)) + ' days.')
