'''One line summary goes here.

Longer description goes here.

    Typical usage example:

    IC = IceColumn
'''

import firedrake

class Stratigrapher:
    '''Track the basal ice stratigraphy beneath an ice mass.

    Intended to operate on one or more IceColumn objects, with functions to handle
    melt, freeze-on, regelation, and frozen fringe development.

    Attributes:
        None

    '''

    def __init__(self, mesh, initial_x: float, initial_y: float, sliding_velocity, ice_thickness,
                 effective_pressure, till_depth, geothermal_flux, conductive_flux):
        '''Initialize the Stratigrapher with input fields from coupled models.'''

        self.mesh = mesh
        self.sliding_velocity = sliding_velocity
        self.ice_thickness = ice_thickness
        self.effective_pressure = effective_pressure
        self.till_depth = till_depth
        self.geothermal_flux = geothermal_flux
        self.conductive_flux = conductive_flux

        # Simulation variables
        self.ice_column = IceColumn(initial_x, initial_y, sliding_velocity, ice_thickness)
        self.time_elapsed = 0.0

        # Constants
        self.array_conductivity = 2e-15 * 1e6 * 3.14e7 # m^2 MPa^-1 a^-1

    def get_attr(self, field):
        '''Returns the value of a field variable at the IceColumn location.'''

        return field.at(self.ice_column.loc['x'], self.ice_column.loc['y'])

    def melt(self, total_melt: float):
        '''Melt the lowest layer(s) of the IceColumn.'''

        to_melt = total_melt

        while to_melt >= self.ice_column.layers[-1].height:
            to_melt -= self.ice_column.layers[-1].height
            self.ice_column.rm_layer(-1)

            if len(self.ice_column.layers) == 0:
                break

        if to_melt >= 0:
            self.ice_column.layers[-1].height -= to_melt
            to_melt = 0.0

    def regeler(self, dt: float):
        '''Entrain a new layer of sediment via regelation.'''

        N = self.get_attr(self.effective_pressure)
        depth = self.ice_column.height
        regelation_rate = self.array_conductivity * (N / depth)

        if self.ice_column.layers[-1].process == 'regelation':
            self.ice_column.layers[-1].height += regelation_rate * dt
        else:
            new_layer = IceLayer(regelation_rate * dt, 0.05, process = 'regelation')
            self.ice_column.add_layer(new_layer)

    def update(self, dt):
        '''Simulate the evolution of the IceColumn over one time step.'''
        pass

class IceColumn:
    '''Data structure for one column of ice.

    Stores multiple IceLayer objects, provides routines to alter some or all of
    the IceLayers within the same column, and stores the total column thickness.

    Attributes:
        None

    '''

    def __init__(self, x: float, y: float, sliding_velocity, ice_thickness):
        '''Initialize the IceColumn with a position, velocity field, and thickness field.'''

        self.loc = {'x': x, 'y': y}
        self.sliding_velocity_field = sliding_velocity
        self.divergence_field = firedrake.div(self.sliding_velocity_field)
        self.ice_thickness = ice_thickness

        # Velocity, divergence, and ice thickness at the current location
        self.sliding_velocity = None
        self.divergence = None
        self.total_height = None
        self.update_field_vars()

        # Simulation variables
        self.height = 0.0
        self.dH_ratio = 1.0
        self.layers = []
        self.time_elapsed = 0.0

    def add_layer(self, new_layer: IceLayer):
        '''Add an IceLayer to the IceColumn.'''

        self.layers.append(new_layer)
        self.height += new_layer.height

    def rm_layer(self, idx: int):
        '''Remove a layer from the IceColumn.'''

        self.height -= self.layers[idx].height
        del self.layers[idx]

    def update_field_vars(self):
        '''Interpolate to find the velocity, divergence, and total height at the current location.'''

        self.sliding_velocity = self.sliding_velocity_field.at(self.loc['x'], self.loc['y'])
        self.divergence = self.divergence_field.at(self.loc['x'], self.loc['y'])
        self.total_height = self.ice_thickness.at(self.loc['x'], self.loc['y'])loc['x'], self.loc['y']])

    def update_loc(self, dt):
        '''Change the position of the IceColumn, given the time step dt.'''

        self.update_velocity()
        ux, uy = self.sliding_velocity
        self.loc['x'] += ux * dt
        self.loc['y'] += uy * dt

    def calc_deformation(self, dt):
        '''Update the expected change in thickness from advective deformation.'''
        dhdt = -self.divergence
        total_dh = dhdt * dt
        self.dh_ratio = self.total_height / (self.total_height + total_dh)

    def update_height(self):
        '''Change the height of the IceColumn and all associated layers.'''

        ratio = self.dh_ratio
        self.height *= ratio

        for idx in range(len(self.layers)):
            new_h = self.layers[idx].height * ratio
            self.layers[idx].mutate(new_h)

    def update(self, dt):
        '''Run one time step, advect the column, and update layer heights.'''

        self.update_field_vars()
        self.update_loc(dt)
        self.calc_deformation(dt)
        self.update_height(dt)
        self.time_elapsed += dt

class IceLayer:
    '''Data structure for one ice layer.'''

    def __init__(self, height: float, concentration: float,
                 angularity = None, grainsize = None, process: string = None):
        '''Initialize the IceLayer with a height and sediment profile.'''

        self.height = height # meters
        self.concentration = concentration # % by vol.
        self.angularity = angularity # 0-1
        self.grainsize = grainsize # distribution or Dx metric
        self.process = process # string

    def mutate(self, new_height: float):
        '''Change the height of the IceLayer by delta_h.'''

        ratio = self.height / new_height
        self.concentration = self.concentration / ratio
        self.height = new_height
