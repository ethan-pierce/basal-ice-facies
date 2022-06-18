'''Class to generate a flowband and calculate boundary conditions at the ice-sediment interface.'''

import numpy as np
import yaml
import rasterio as rio
from landlab import RasterModelGrid
from dataclasses import dataclass

@dataclass
class Flowline:
    '''Dataclass stores values of fields along a flowline.'''
    node_id: np.ndarray
    node_x: np.ndarray
    node_y: np.ndarray
    distance: np.ndarray
    fields: dict

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

        self.break_nodes = np.where(self.grid.at_node[inputs['break_condition']['field']] <=
                                    inputs['break_condition']['min_value'])[0]

    def generate_flowline(self, dt: float, max_iter: int, omit: list = []):
        '''Generate an experimental flowline from a random (valid) starting location.'''

        node_id = []
        node_x = []
        node_y = []
        distance = []
        fields = {var: [] for var in self.variables if var not in omit}

        start_node = np.random.choice(self.initial_nodes)
        origin_x = self.grid.node_x[start_node]
        origin_y = self.grid.node_y[start_node]
        current_x = self.grid.node_x[start_node]
        current_y = self.grid.node_y[start_node]

        node = self.grid.find_nearest_node((current_x, current_y))

        for i in range(max_iter):
            node_id.append(node)
            node_x.append(self.grid.node_x[node])
            node_y.append(self.grid.node_y[node])
            distance.append(np.sqrt((node_x[-1] - origin_x)**2 + (node_y[-1] - origin_y)**2))

            for var in fields.keys():
                fields[var].append(self.grid.at_node[var][node])

            current_x += self.grid.at_node['velocity_x'] * (1 / self.resolution[0]) * dt
            current_y += self.grid.at_node['velocity_y'] * (1 / self.resolution[1]) * dt

            try:
                node = self.grid.find_nearest_node((current_x, current_y))
            except:
                break

            if node in self.break_nodes:
                break

        flowline = Flowline(node_id = node_id,
                            node_x = node_x,
                            node_y = node_y,
                            distance = distance,
                            fields = fields)

        return flowline

    def construct_flowband(self, surface: str, dt: float, max_iter: int, omit: list = []):
        '''Construct a flowband from an existing flowline.'''

        flowline = self.generate_flowline(dt, max_iter, omit)

        if surface not in flowline.fields.keys():
            raise ValueError('Missing ' + surface + ' field from flowline.')

        flowband = RasterModelGrid((len(flowline.distance), np.max(flowline.fields[surface]) + 1))
