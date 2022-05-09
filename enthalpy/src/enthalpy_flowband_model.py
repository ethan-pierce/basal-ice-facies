import numpy as np
from typing import Callable

class EnthalpyFlowband:

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

    def __init__(self, nx: int, nz: int, dx: float, dz: float,
                 initial_temperature: float,
                 surface_function: Callable[np.ndarray, np.ndarray],
                 velocity_function: Callable,
                 temperature_boundary: np.ndarray,
                 porosity_boundary: np.ndarray,
                 start_temperate = False):

        # Construct grid
        self.nx = nx
        self.nz = nz
        self.shape = (nx, nz)
        self.dx = dx
        self.dz = dz
        self.xs = np.arange(0, self.nx * self.dx, self.dx)
        self.zs = np.arange(0, self.nz * self.dz, self.dz)
        self.xgrid, self.zgrid = np.meshgrid(self.xs, self.zs)
        self.zidx, self.xidx = np.indices(self.shape)

        # Allocate fields for initial conditions
        empty = np.empty(self.shape)
        self.initial_temperature = initial_temperature
        self.surface_function = surface_function
        self.velocity_function = velocity_function
        self.surface = self.surface_function(self.xgrid)
        self.in_glacier = np.where(self.zgrid < self.surface, 1, 0)

        self.at_surface = empty
        for row in range(self.nz):
            for col in range(self.nx):
                if self.in_glacier[row, col] == 1 and row != 0:
                    if self.in_glacier[row + 1, col] == 0:
                        self.at_surface[row, col] = 1
                else:
                    self.at_surface[row, col] = 0

        self.hydrostatic_pressure = self.rho_ice * self.gravity * (self.surface - self.zgrid)

        self.melt_point = self.melt_point = (
            self.temperature_ref - self.clapeyron * (self.hydrostatic_pressure - self.pressure_ref)
        )

        self.velocity_x = self.velocity_function(self.xgrid, self.zgrid, self.surface)
        gradient_velocity_x = self.calc_dx(self.velocity_x)
        self.velocity_z = -gradient_velocity_x * self.zgrid

        self.effective_viscosity = empty
        self.internal_energy = empty
        self.calc_internal_energy()

        # Store simulation variables
        self.temperature_boundary = temperature_boundary
        self.porosity_boundary = porosity_boundary
        self.time_elapsed = 0.0
        self.advection = empty
        self.diffusion = empty
        self.water_flux = empty
        self.enthalpy_dt = empty

        # Allocate model fields
        self.is_temperate = np.full(self.shape, 0)

        if start_temperate:
            self.temperature = self.melt_point.copy()
        else:
            self.temperature = np.full(self.shape, self.initial_temperature)
            
        self.porosity = np.zeros(self.shape)
        self.enthalpy = self.calc_enthalpy(self.temperature, self.porosity)
        self.effective_pressure = self.hydrostatic_pressure.copy()
        self.partition_domain()

    def calc_dx(self, field: np.ndarray) -> np.ndarray:
        return np.gradient(field, self.dx, axis = 1)

    def calc_dz(self, field: np.ndarray) -> np.ndarray:
        return np.gradient(field, self.dz, axis = 0)

    def calc_enthalpy(self, temperature: np.ndarray, porosity: np.ndarray) -> np.ndarray:
        enthalpy = (self.rho_ice * self.heat_capacity * (temperature - self.temperature_ref)
                    + self.rho_water * self.latent_heat * porosity)
        return enthalpy

    def update_temperature(self):
        cold_ice_temperature = self.temperature_ref + self.enthalpy / (self.rho_ice * self.heat_capacity)
        self.temperature = np.where(self.is_temperate, self.melt_point, cold_ice_temperature)
        self.temperature = np.where(self.temperature > self.melt_point, self.melt_point, self.temperature)

    def update_porosity(self):
        porosity_if_temperate = ((self.enthalpy - (self.rho_ice * self.heat_capacity *
                                 (self.melt_point - self.temperature_ref))) /
                                 (self.rho_water * self.latent_heat))
        self.porosity = np.where(self.is_temperate, porosity_if_temperate, 0.0)

    def update_pressure(self):
        effective_pressure = self.effective_pressure.copy()

        # Enforce effective pressure = 0 at the surface
        baseline_pressure = np.where(self.at_surface == 1, 0.0, effective_pressure)
        pressure_gradient = self.calc_dz(baseline_pressure)

        # Enforce d(effective pressure) / dz = 0 at the base
        pressure_gradient[self.nz - 1, :] = 0.0
        inner_function = ((self.rho_water - self.rho_ice) * self.gravity + pressure_gradient)
        laplacian = self.calc_dz(inner_function)
        diffusivity = self.effective_viscosity * self.drainage_coeff / self.water_viscosity
        compaction_pressure = diffusivity * laplacian

        with np.errstate(divide = 'ignore'):
            self.effective_pressure[:] = np.where(self.porosity > 0,
                                                  compaction_pressure / self.porosity,
                                                  self.hydrostatic_pressure)

    def set_initial_conditions(self):

        # Set glacier surface
        self.surface = self.surface_function(self.xgrid)
        self.in_glacier = np.where(self.zgrid < self.surface, 1, 0)

        for row in range(self.nz):
            for col in range(self.nx):
                if self.in_glacier[row, col] == 1 and row != 0:
                    if self.in_glacier[row + 1, col] == 0:
                        self.at_surface[row, col] = 1
                else:
                    self.at_surface[row, col] = 0

        for row in range(self.nz):
            for col in range(self.nx):
                if self.at_surface[row, col] == 1:
                    self.near_surface[row - 1, col] = 1

        # Set pressure melting point
        self.hydrostatic_pressure = self.rho_ice * self.gravity * (self.surface - self.zgrid)
        self.melt_point = (self.temperature_ref - self.clapeyron *
                           (self.hydrostatic_pressure - self.pressure_ref))

        # Set initial velocity
        self.velocity_x = self.velocity_function(self.xgrid, self.zgrid, self.surface)
        gradient_velocity_x = self.calc_dx(self.velocity_x)
        self.velocity_z = -gradient_velocity_x * self.zgrid

    def calc_internal_energy(self):

        du_dx = self.calc_dx(self.velocity_x)
        du_dz = self.calc_dz(self.velocity_x)
        dw_dx = self.calc_dx(self.velocity_z)
        dw_dz = self.calc_dz(self.velocity_z)

        # Convention: store tensor components in C-style (row-major) ordering, such that
        # index 0 is xx or du/dx, 1 is xz or du/dz, 2 is zx or dw/dx, and 3 is zz or dw/dz
        strain_rate_xx = du_dx
        strain_rate_xz = du_dz + dw_dx
        strain_rate_zz = dw_dz

        effective_strain = np.sqrt(
                               0.5 * (strain_rate_xx**2 + 2 * strain_rate_xz**2 + strain_rate_zz**2)
                           )

        self.effective_viscosity = (self.glens_coeff**(-1 / self.glens_n) *
                                    effective_strain**(1 / (self.glens_n - 1)))

        self.internal_energy = np.sum(
            [self.effective_viscosity * strain_rate_xx**2,
             2 * self.effective_viscosity * strain_rate_xz**2,
             self.effective_viscosity * strain_rate_zz**2],
             axis = 0
        )

    def partition_domain(self):
        threshold = self.rho_ice * self.heat_capacity * (self.melt_point - self.temperature_ref)
        self.is_temperate = np.where(self.enthalpy >= threshold, 1, 0)

    def compute_advection(self):
        inflow_enthalpy = self.calc_enthalpy(self.temperature_boundary[2], self.porosity_boundary[2])
        outflow_enthalpy = self.calc_enthalpy(self.temperature_boundary[3], self.porosity_boundary[3])
        self.enthalpy = np.where(self.xidx == 0, inflow_enthalpy, self.enthalpy)
        self.enthalpy = np.where(self.xidx == np.max(self.xidx), outflow_enthalpy, self.enthalpy)

        enthalpy_dx = self.calc_dx(self.enthalpy)
        enthalpy_dz = self.calc_dz(self.enthalpy)

        self.advection = self.velocity_x * enthalpy_dx + self.velocity_z * enthalpy_dz

    def compute_diffusion(self):
        surface_temperature = self.temperature_boundary[0]
        self.temperature = np.where(self.at_surface == 1, surface_temperature, self.temperature)
        self.temperature = np.where(self.zidx == np.min(self.zidx), self.melt_point, self.temperature)

        temperature_dz = self.calc_dz(self.temperature)
        self.diffusion = self.calc_dz(self.conductivity * temperature_dz)

    def compute_water_flux(self):
        flux_coeff = self.rho_water * self.latent_heat
        water_flux_coeff = ((self.drainage_coeff * self.porosity**self.drainage_exponent) / self.water_viscosity)
        pressure_dz = self.calc_dz(self.effective_pressure)
        self.water_flux = water_flux_coeff * ((self.rho_water - self.rho_ice) * self.gravity + pressure_dz)

    def compute_enthalpy_slope(self) -> np.ndarray:
        enthalpy_slope = (self.internal_energy + self.diffusion - self.advection - self.water_flux)
        return enthalpy_slope

    def clean_up_out_of_bounds(self):
        self.enthalpy[:] *= self.in_glacier
        self.temperature[:] *= self.in_glacier
        self.porosity[:] *= self.in_glacier
        self.effective_pressure[:] *= self.in_glacier

    def run_one_step(self, dt: float):

        # Solve the enthalpy conservation equation
        self.partition_domain()
        self.compute_advection()
        self.compute_diffusion()
        self.compute_water_flux()
        self.enthalpy_dt[:] = self.compute_enthalpy_slope()
        self.enthalpy[:] = self.enthalpy_dt * dt

        # Update state variables
        self.update_temperature()
        self.update_porosity()
        self.update_pressure()
        self.time_elapsed += dt

        self.clean_up_out_of_bounds()
