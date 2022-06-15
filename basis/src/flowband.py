'''Class to generate a flowband and calculate boundary conditions at the ice-sediment interface.'''

import numpy as np
from dataclasses import dataclass
from landlab import RasterModelGrid
import rasterio as rio

class FlowbandGenerator:
    '''Class to generate a Lagrangian flowband and populate grid fields at nodes.'''

    def __init__(self, config: str):
        '''Initialize the utility with a configuration file.'''

        self.variables = []
        self.crs = None
        self.grid = None
