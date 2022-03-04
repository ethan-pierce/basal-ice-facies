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

    def __init__(self):
        '''Initialize the Stratigrapher with fields from an ice dynamics model.'''
        pass

    def erode(self):
        '''Calculate subglacial erosion beneath the ice mass.'''
        pass

    def melt(self):
        '''Melt the lowest layer(s) of the IceColumn.'''
        pass

    def freeze(self):
        '''Freeze on a new layer at the base of the IceColumn.'''
        pass

    def regeler(self):
        '''Entrain a new layer of sediment via regelation.'''
        pass

    def evolve_fringe(self):
        '''Update the state of the frozen fringe beneath the ice mass.'''
        pass

    def update_thermal_state(self):
        '''Update the thermal state at the base of the ice mass.'''
        pass

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

    def update_layer(self, idx, **kwargs):
        '''Change the attributes of a layer.'''

        for key, val in kwargs:
            setattr(self.layers[idx], key, val)

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
            self.update_layer(idx, height = new_h)

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
                 angularity = None, grainsize = None):
        '''Initialize the IceLayer with a height and sediment profile.'''

        self.height = height # meters
        self.concentration = concentration # % by vol.
        self.angularity = angularity # 0-1
        self.grainsize = grainsize # distribution or Dx metric

    def mutate(self, delta_h):
        '''Change the height of the IceLayer by delta_h.'''

        ratio = self.height / (self.height + delta_h)
        self.concentration = self.concentration / ratio
        self.height += delta_h
