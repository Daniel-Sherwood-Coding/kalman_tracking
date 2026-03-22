import numpy as np
from mesa.discrete_space import CellAgent


class PredictedTargetAgent(CellAgent):
    """A non-moving marker agent for SearcherAgent's predicted target location."""

    def __init__(self, model, cell, searcher, uncertainty=0.0):
        super().__init__(model)
        self.cell = cell
        self.searcher = searcher
        self.uncertainty = float(uncertainty)

    def get_position(self):
        return np.asarray(self.cell.position, dtype=float)

    def step(self):
        pass