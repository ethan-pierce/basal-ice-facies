'''Class to model enthalpy along a flowband.'''

import numpy as np
from landlab import RasterModelGrid

class CompactionPressureModel:
    '''Uses a compaction-pressure model as a closure term, following Hewitt and Schoof (2017).'''

    def __init__(self, mesh: RasterModelGrid):
        '''Initialize the model with a Flowband object.'''
        self.mesh = mesh
