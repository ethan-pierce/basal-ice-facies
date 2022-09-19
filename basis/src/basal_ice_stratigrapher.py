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

        for key in inputs['static_inputs'].keys():
            if key not in self.grid.at_node.keys():
                self.grid.add_zeros(key, at = 'node', units = inputs['static_inputs'][key]['units'])
                self.grid.at_node[key][:] = inputs['static_inputs'][key]['value']

        for key in inputs['input_fields'].keys():
            if key not in self.grid.at_node.keys():
                with rio.open(inputs['input_fields'][key][file], 'r') as f:
                    data = f.read(1)

                    if data.shape == self.grid.shape:
                        self.grid.add_field(key, data, at = 'node', units = inputs['input_fields'][key]['units'])

                    else:
                        raise ValueError("Shape of " + str(key) + " data does not match grid shape.")

        self.grid.status_at_node[self.grid.at_node['glacier__thickness'][:] <= 0] = self.grid.BC_NODE_IS_CLOSED
        self.calc_thermal_gradients()

    def calc_erosion_rate(self):
        """Calculates the erosion rate as a function of sliding velocity."""
        required = ['soil__depth', 'glacier__sliding_velocity_x', 'glacier__sliding_velocity_y']

        for field in required:
            if field not in self.grid.at_node.keys():
                raise ValueError("Missing " + str(field) + " at nodes.")

        if 'erosion__rate' not in self.grid.at_node.keys():
            self.grid.add_zeros('erosion__rate', at = 'node')

        Ks = self.parameters['erosion_coefficient']
        m = self.parameters['erosion_exponent']

        ub = np.sqrt(
            self.grid.at_node['glacier__sliding_velocity_x'][:]**2 +
            self.grid.at_node['glacier__sliding_velocity_y'][:]**2
        )

        self.grid.at_node['erosion__rate'][:] = (
            Ks * np.abs(ub)**m
        )

    def calc_melt_rate(self):
        """Calculates the basal melt rate as a function of shear stress and heat fluxes."""
        required = ['glacier__sliding_velocity_x', 'glacier__sliding_velocity_y',
                    'glacier__effective_pressure']

        for field in required:
            if field not in self.grid.at_node.keys():
                raise ValueError("Missing " + str(field) + " at nodes.")

        if 'subglacial_melt__rate' not in self.grid.at_node.keys():
            self.grid.add_zeros('subglacial_melt__rate', at = 'node')

        if 'frictional_heat__flux' not in self.grid.at_node.keys():
            self.grid.add_zeros('frictional_heat__flux', at = 'node')

        rho = self.parameters['ice_density']
        L = self.parameters['ice_latent_heat']

        ub = np.sqrt(
            self.grid.at_node['glacier__sliding_velocity_x'][:]**2 +
            self.grid.at_node['glacier__sliding_velocity_y'][:]**2
        )

        N = self.grid.at_node['glacier__effective_pressure'][:]
        mu = self.parameters['friction_coefficient']

        frictional = ub * mu * N
        self.grid.at_node['frictional_heat__flux'][:] = frictional

        geothermal = self.parameters['geothermal_heat_flux']

        self.grid.at_node['subglacial_melt__rate'][:] = (frictional + geothermal) / (rho * L)

    def calc_thermal_gradients(self):
        """Calculates the temperature gradients through the frozen fringe and dispersed layer."""
        required = []

        for field in required:
            if field not in self.grid.at_node.keys():
                raise ValueError("Missing " + str(field) + " at nodes.")

        if 'fringe__thermal_gradient' not in self.grid.at_node.keys():
            self.grid.add_zeros('fringe__thermal_gradient', at = 'node')

        if 'frozen_fringe__thickness' not in self.grid.at_node.keys():
            self.grid.add_zeros('frozen_fringe__thickness', at = 'node')

        # Calculate entry pressure
        if 'entry_pressure' not in self.parameters.keys():
            gamma = self.parameters['surface_energy']
            rp = self.parameters['pore_throat_radius']
            self.parameters['entry_pressure'] = (2 * gamma) / rp

        # Calculate temperature at base of fringe
        if 'fringe_base_temperature' not in self.parameters.keys():
            pf = self.parameters['entry_pressure']
            Tm = self.parameters['melt_temperature']
            rho = self.parameters['ice_density']
            L = self.parameters['ice_latent_heat']
            self.parameters['fringe_base_temperature'] = Tm - ((pf * Tm) / (rho * L))

        if 'fringe_conductivity' not in self.parameters.keys():
            ki = self.parameters['ice_thermal_conductivity']
            ks = self.parameters['sediment_thermal_conductivity']
            phi = self.parameters['frozen_fringe_porosity']
            self.parameters['fringe_conductivity'] = (1 - phi) * ks + phi * ki

        if 'frictional_heat__flux' not in self.grid.at_node.keys():
            self.calc_melt_rate()

        K = self.parameters['fringe_conductivity']
        Qg = self.parameters['geothermal_heat_flux']
        Qf = self.grid.at_node['frictional_heat__flux'][:]
        self.grid.at_node['fringe__thermal_gradient'][:] = -(Qg + Qf) / K

        G = self.grid.at_node['fringe__thermal_gradient'][:]
        hf = self.grid.at_node['frozen_fringe__thickness'][:]
        Tf = self.parameters['fringe_base_temperature']

        self.grid.at_node['transition_temperature'] = G * hf + Tf

    def calc_fringe_growth_rate(self):
        """Calculates the growth rate of frozen fringe as a function of melt and pressure."""
        required = ['glacier__effective_pressure']

        for field in required:
            if field not in self.grid.at_node.keys():
                raise ValueError("Missing " + str(field) + " at nodes.")

        if 'fringe__undercooling' not in self.grid.at_node.keys():
            self.grid.add_zeros('fringe__undercooling', at = 'node')

        if 'fringe__saturation' not in self.grid.at_node.keys():
            self.grid.add_zeros('fringe__saturation', at = 'node')

        if 'nominal__heave_rate' not in self.grid.at_node.keys():
            self.grid.add_zeros('nominal__heave_rate', at = 'node')

        if 'flow__resistance' not in self.grid.at_node.keys():
            self.grid.add_zeros('flow__resistance', at = 'node')

        G = self.grid.at_node['fringe__thermal_gradient'][:]
        h = self.grid.at_node['frozen_fringe__thickness'][:]
        Tm = self.parameters['melt_temperature']
        Tf = self.parameters['fringe_base_temperature']
        self.grid.at_node['fringe__undercooling'][:] = 1 - ((G * h) / (Tm - Tf))

        alpha = self.parameters['fringe_alpha']
        beta = self.parameters['fringe_beta']
        theta = self.grid.at_node['fringe__undercooling'][:]
        self.grid.at_node['fringe__saturation'][:] = 1 - theta**(-beta)

        rho_w = self.parameters['water_density']
        rho_i = self.parameters['ice_density']
        L = self.parameters['ice_latent_heat']
        k0 = self.parameters['permeability']
        eta = self.parameters['water_viscosity']
        self.grid.at_node['nominal__heave_rate'][:] = -(rho_w**2 * L * G * k0) / (rho_i * Tm * eta)

        d = self.parameters['film_thickness']
        R = self.parameters['till_grain_radius']
        self.grid.at_node['flow__resistance'][:] = -(rho_w**2 * k0 * G * R**2) / (rho_i**2 * (Tm - Tf) * d**3)

        phi = self.parameters['frozen_fringe_porosity']
        Vs = self.grid.at_node['nominal__heave_rate'][:]
        Pi = self.grid.at_node['flow__resistance'][:]
        N = self.grid.at_node['glacier__effective_pressure'][:]
        pf = self.parameters['entry_pressure']

        # Throwaway variables for long coefficients
        A = theta + phi * (1 - theta + (1 / (1 - beta)) * (theta**(1 - beta) - 1))
        B = ((1 - phi)**2 / (alpha + 1)) * (theta**(alpha + 1) - 1)
        C = ((2 * (1 - phi) * phi) / (alpha - beta + 1)) * (theta**(alpha - beta + 1) - 1)
        D = (phi**2 / (alpha - 2 * beta + 1)) * (theta**(alpha - 2 * beta + 1) - 1)

        self.grid.at_node['fringe__heave_rate'][:] = Vs * (A - (N / pf)) / (B + C + D + Pi)
        V = self.grid.at_node['fringe__heave_rate'][:]
        m = self.grid.at_node['subglacial_melt__rate'][:]
        S = self.grid.at_node['fringe__saturation'][:]

        self.grid.at_node['fringe__growth_rate'] = np.where(
            S > 0,
            (-m - V) / (phi * S),
            0.0
        )

    def calc_regelation_rate(self):
        """Calculates the vertical regelation rate and change in depth-averaged sediment concentration."""
        required = ['fringe__undercooling']

        for field in required:
            if field not in self.grid.at_node.keys():
                raise ValueError("Missing " + str(field) + " at nodes.")

        if 'dispersed_layer__gradient' not in self.grid.at_node.keys():
            self.grid.add_zeros('dispersed_layer__gradient', at = 'node')

        if 'dispersed_layer__growth_rate' not in self.grid.at_node.keys():
            self.grid.add_zeros('dispersed_layer__growth_rate', at = 'node')

        if 'dispersed_layer__thickness' not in self.grid.at_node.keys():
            self.grid.add_zeros('dispersed_layer__thickness', at = 'node')

        pf = self.parameters['entry_pressure']
        Tm = self.parameters['melt_temperature']
        Tf = self.parameters['fringe_base_temperature']
        rho = self.parameters['ice_density']
        g = self.parameters['gravity']
        L = self.parameters['ice_latent_heat']
        r = self.parameters['particle_radius']
        z0 = self.parameters['critical_depth']
        gamma = self.parameters['ice_clapeyron_slope']
        theta = self.grid.at_node['fringe__undercooling'][:]

        # Meyer et al. 2018, eq (12)
        temp_at_top_of_fringe = Tm - (Tm - Tf) * theta

        # The temperature gradient depends on the supercooling at the top of the fringe and the pressure-melting-point
        G_premelting = (Tm - temp_at_top_of_fringe) / z0
        G_pressure_melting = gamma * rho * g
        G = G_premelting + G_pressure_melting
        self.grid.at_node['dispersed_layer__gradient'][:] = G

        # Kozeny-Carmen equation for permeability
        phi = self.parameters['cluster_volume_fraction']
        K = (r**2 * (1 - phi)**3) / (45 * phi**2)

        mu = self.parameters['water_viscosity']
        ki = self.parameters['ice_thermal_conductivity']
        kp = self.parameters['sediment_thermal_conductivity']
        coeff = (K * rho * L) / (mu * Tm * (2 * ki + kp))

        # The dispersed layer thickness grows with the velocity of the fastest particles
        self.grid.at_node['dispersed_layer__growth_rate'][:] = (coeff * 3 * ki) / (1 + coeff * rho * L) * G

    def calc_advection(self):
        """Calculates transport of basal ice layers by horizontal advection. This scheme is likely to be unstable."""
        required = ['glacier__sliding_velocity_x', 'glacier__sliding_velocity_y',
                    'fringe__growth_rate', 'frozen_fringe__thickness',
                    'dispersed_layer__growth_rate', 'dispersed_layer__thickness']

        for field in required:
            if field not in self.grid.at_node.keys():
                raise ValueError("Missing " + str(field) + " at nodes.")

        if 'dispersed_layer__advection' not in self.grid.at_node.keys():
            self.grid.add_zeros('dispersed_layer__advection', at = 'node')

        if 'frozen_fringe__advection' not in self.grid.at_node.keys():
            self.grid.add_zeros('frozen_fringe__advection', at = 'node')

        hd = self.grid.at_node['dispersed_layer__thickness'][:]
        hf = self.grid.at_node['frozen_fringe__thickness'][:]
        u = self.grid.at_node['glacier__sliding_velocity_x'][:]
        v = self.grid.at_node['glacier__sliding_velocity_y'][:]

        grad_hd = self.grid.calc_grad_at_link(hd)
        hd_x = self.grid.map_mean_of_horizontal_active_links_to_node(grad_hd)
        hd_y = self.grid.map_mean_of_vertical_active_links_to_node(grad_hd)
        advect_hd = u * hd_x + v * hd_y
        self.grid.at_node['dispersed_layer__advection'][:] = advect_hd
        self.grid.at_node['dispersed_layer__growth_rate'][:] -= advect_hd

        grad_hf = self.grid.calc_grad_at_link(hf)
        hf_x = self.grid.map_mean_of_horizontal_active_links_to_node(grad_hf)
        hf_y = self.grid.map_mean_of_vertical_active_links_to_node(grad_hf)
        advect_hf = u * hf_x + v * hf_y
        self.grid.at_node['frozen_fringe__advection'][:] = advect_hf
        self.grid.at_node['fringe__growth_rate'][:] -= advect_hf

    def calc_dynamic_thinning(self):
        """Calculates the dynamic thickening or thinning of basal ice layers."""
        required = ['glacier__sliding_velocity_x', 'glacier__sliding_velocity_y']

        for field in required:
            if field not in self.grid.at_node.keys():
                raise ValueError("Missing " + str(field) + " at nodes.")

        if 'glacier__velocity_divergence' not in self.grid.at_node.keys():
            u = self.grid.at_node['glacier__sliding_velocity_x'][:]
            v = self.grid.at_node['glacier__sliding_velocity_y'][:]

            grad_u = self.grid.calc_grad_at_link(u)
            grad_v = self.grid.calc_grad_at_link(v)

            du_dx = self.grid.map_mean_of_horizontal_active_links_to_node(grad_u)
            dv_dy = self.grid.map_mean_of_vertical_active_links_to_node(grad_v)

            self.grid.add_field('glacier__velocity_divergence', du_dx + dv_dy, at = 'node')

        self.grid.at_node['fringe__growth_rate'][:] -= (
            self.grid.at_node['frozen_fringe__thickness'][:] * self.grid.at_node['glacier__velocity_divergence'][:]
        )

        self.grid.at_node['dispersed_layer__growth_rate'][:] -= (
            self.grid.at_node['dispersed_layer__thickness'][:] * self.grid.at_node['glacier__velocity_divergence']
        )

    def run_one_step(self, dt, advection = True, dynamic_thinning = True):
        """Advances the model forward one time step of size dt (seconds)."""

        # TODO split operators between source terms and advection terms

        self.calc_erosion_rate()
        self.grid.at_node['soil__depth'][:] += self.grid.at_node['erosion__rate'][:] * dt

        self.calc_melt_rate()
        self.calc_thermal_gradients()
        self.calc_fringe_growth_rate()

        # TODO what happens when fringe growth > soil depth?

        self.calc_regelation_rate()

        if advection == True:
            self.calc_advection()

        if dynamic_thinning == True:
            self.calc_dynamic_thinning()

        self.grid.at_node['dispersed_layer__growth_rate'][self.grid.at_node['glacier__thickness'] <= 0] = 0.0
        self.grid.at_node['fringe__growth_rate'][self.grid.at_node['glacier__thickness'] <= 0] = 0.0

        self.grid.at_node['frozen_fringe__thickness'][:] += (
            self.grid.at_node['fringe__growth_rate'][:] * dt
        )

        self.grid.at_node['frozen_fringe__thickness'][:] = np.where(
            self.grid.at_node['frozen_fringe__thickness'][:] >= 0.0,
            self.grid.at_node['frozen_fringe__thickness'][:],
            0.0
        )

        self.grid.at_node['dispersed_layer__thickness'][:] += (
            self.grid.at_node['dispersed_layer__growth_rate'] * dt
        )

        # TODO how much mass does the fringe lose to the dispersed layer? Assume negligible effect on hf?
