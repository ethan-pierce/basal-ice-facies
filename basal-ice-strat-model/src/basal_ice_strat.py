'''One line summary goes here.

Longer description goes here.

    Typical usage example:

    IC = IceColumn
'''

import firedrake

class IceColumn:
    '''Data structure for one column of ice.

    Stores multiple IceLayer objects, provides routines to alter some or all of
    the IceLayers within the same column, and stores the total column thickness.

    Attributes:
        None

    '''

    def __init__(self, x: float, y: float, flow_field):
        '''Initialize the IceColumn with a position and a velocity field.'''
        
        self.loc = {'x': x, 'y': y}
        self.flow_field = flow_field
        self.div_field = firedrake.div(self.flow_field)

        # Velocity and divergence at the current location
        self.velocity = None
        self.divergence = None

        # Simulation variables
        self.height = 0.0
        self.layers = []
        self.time_elapsed = 0.0

    def add_layer(self, new_layer: IceLayer):
        '''Add an IceLayer to the IceColumn.'''

        self.layers.append(new_layer)
        pass

    def rm_layer(self):
        '''Remove a layer from the IceColumn.'''
        pass

    def mutate_layer(self):
        '''Change the attributes of a layer.'''
        pass

    def interpolate(self):
        '''Interpolate the column position within the velocity field.'''
        pass

    def move(self):
        '''Change the position of the IceColumn.'''
        pass

    def alter_height(self):
        '''Change the height of the IceColumn and all associated layers.'''
        pass

    def update(self, dt):
        '''Run one time step, advect the column, and update layer heights.'''
        pass


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
