'''Class to generate a flowband and calculate boundary conditions at the ice-sediment interface.'''

import numpy as np
import yaml
import rasterio as rio
from landlab import RasterModelGrid

class FlowbandGenerator:
    '''Class to generate a Lagrangian flowband and populate grid fields at nodes.'''

    def __init__(self, config_file: str):
        '''Initialize the utility with a configuration file.'''

        with open(config_file, 'r') as config:
            inputs = yaml.safe_load(config)

        self.tag = inputs['tag']
        self.user = inputs['user']
        self.date = inputs['date']
        self.template = inputs['template']
        self.variables = inputs['fields'].keys()
        self.grid = None

        with rio.open(inputs['fields'][self.template]['file']) as file:
            self.crs = file.crs
            self.shape = file.read(1).shape
            self.lower_left = (file.bounds.left, file.bounds.bottom)
            self.resolution = file.res
            self.grid = RasterModelGrid(self.shape, xy_of_lower_left = self.lower_left, xy_spacing = self.resolution)

        if self.grid is None:
            raise AttributeError('No template file specified, could not create RasterModelGrid.')

        for var in self.variables:
            metadata = inputs['fields'][var]

            with rio.open(metadata['file']) as file:
                if metadata['flip_axes'] is not None:
                    data = metadata['scale_by'] * np.flip(file.read(1), axis = metadata['flip_axes'])
                else:
                    data = metadata['scale_by'] * file.read(1)

                if data.shape != self.shape:
                    raise ValueError('Shape of ' + var + ' is ' + str(data.shape) + ', but should be ' + str(self.shape))

                self.grid.add_field(var, data, at = metadata['at'], units = metadata['units'])

        self.init_field = inputs['initial_position']['field']
        self.init_min = inputs['initial_position']['min_value']
        self.init_max = inputs['initial_position']['max_value']
        self.init_condition = ((self.grid.at_node[self.init_field] > self.init_min) &
                              (self.grid.at_node[self.init_field] < self.init_max))
                              
        self.initial_nodes = np.where(self.init_condition)[0]

    def generate_flowline(self, dt: float, omit: list = []):
        '''Generate an experimental flowline from a random (valid) starting location.'''
        pass

    def construct_flowband(self):
        '''Construct a flowband from an existing flowline.'''
        pass
