'''Class to generate a flowband and calculate boundary conditions at the ice-sediment interface.'''

from dataclasses import dataclass
from landlab import RasterModelGrid
import rasterio as rio

@dataclass
class Flowband:
    '''Dataclass that wraps a 2D RasterModelGrid in the x-z plane.'''
    grid: RasterModelGrid

class FlowbandGenerator:
    '''Class to generate a Lagrangian flowband and populate grid fields.'''

    def __init__(self, files: list):
        '''Initialize the utility with a list of input files.'''

        self.variables = []

        # Read each file and store the first band of raster data in this class.
        for file in files:
            file_name = file.split('/')[-1]
            var_name = file_name.split('.')[0]
            self.variables.append(var_name)

            with rio.open(file) as f:
                data = f.read(1)

                setattr(self, var_name, data)

        # Set up the 2D x-y mesh
        self.shape = getattr(self, self.variables[0]).shape
        self.grid = RasterModelGrid(self.shape)

        # Assert that all variable fields have the same shape as the first variable listed
        for var in self.variables:
            if getattr(self, var).shape != self.shape:
                raise ValueError(var + ' has shape ' + str(getattr(self, var).shape) + ', but should be ' + str(self.shape))
