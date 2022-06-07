'''Class to generate a flowband and calculate boundary conditions at the ice-sediment interface.'''

from dataclasses import dataclass
from landlab import RasterModelGrid

@dataclass
class Flowband:
    grid: RasterModelGrid
    

class FlowbandGenerator:

    def __init__(self, mesh, sliding_velocity, ice_thickness, effective_pressure):
        pass
