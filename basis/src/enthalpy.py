'''Class to model enthalpy along a flowband.'''
from src.flowband import Flowband

class CompactionPressureModel:
    '''Uses a compaction-pressure model as a closure term, following Hewitt and Schoof (2017).'''

    def __init__(self, flowband: Flowband):
        '''Initialize the model with a Flowband object.'''
        self.flowband = flowband
