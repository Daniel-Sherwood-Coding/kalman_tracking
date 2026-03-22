import numpy as np
from mesa.discrete_space import CellAgent
from .target_agent import TargetAgent


class ReserveSearcherAgent(CellAgent):
    """A support searcher that can deploy from base, search around predicted target, and return."""

    def __init__(self, model, cell, base_position=(500, 999), velocity=8, search_radius=40, search_duration=30):
        super().__init__(model)
        self.cell = cell
        self.base_position = np.asarray(base_position, dtype=float)
        self.velocity = velocity
        self.search_radius = search_radius
        self.search_duration = search_duration

        self.mode = "idle"  # idle, deploy, search, return
        self.search_center = None
        self.remaining_search = 0
        self.last_known_target_position = None

    def _distance_to(self, position):
        return np.linalg.norm(np.asarray(self.cell.position, dtype=float) - np.asarray(position, dtype=float))

    def _move_towards(self, position):
        neighbors = self.cell.get_neighborhood(radius=self.velocity)
        if len(neighbors) == 0:
            return
        target = np.asarray(position, dtype=float)
        best = min(
            neighbors.cells,
            key=lambda c: np.linalg.norm(np.asarray(c.position, dtype=float) - target),
        )
        self.cell = best

    def task_search(self, target_position):
        self.search_center = np.asarray(target_position, dtype=float)
        self.mode = "deploy"
        self.remaining_search = self.search_duration

    def _search_sweep(self):
        # wind around the search center by choosing neighbors in radius with randomization
        neighbors = self.cell.get_neighborhood(radius=self.velocity)
        valid = [c for c in neighbors.cells if np.linalg.norm(np.asarray(c.position, dtype=float) - self.search_center) <= self.search_radius]
        if not valid:
            valid = neighbors.cells
        self.cell = self.model.random.choice(valid)

    def _sense_for_target(self):
        targets = [a for a in self.model.agents if isinstance(a, TargetAgent)]
        for t in targets:
            dist = np.linalg.norm(np.asarray(self.cell.position, dtype=float) - np.asarray(t.cell.position, dtype=float))
            if dist <= self.search_radius:
                measurement = np.asarray(t.cell.position, dtype=float)
                self.model.broadcast_target_detection(measurement)
                return True
        return False

    def step(self):
        if self.mode == "idle":
            return

        if self.mode == "deploy":
            self._move_towards(self.search_center)
            if self._distance_to(self.search_center) <= self.velocity:
                self.mode = "search"
                self.remaining_search = self.search_duration
            return

        if self.mode == "search":
            found = self._sense_for_target()
            if found:
                self.mode = "return"
                return

            self._search_sweep()
            self.remaining_search -= 1
            if self.remaining_search <= 0:
                self.mode = "return"
            return

        if self.mode == "return":
            self._move_towards(self.base_position)
            if self._distance_to(self.base_position) <= self.velocity:
                self.cell = self.model.grid.find_nearest_cell(self.base_position)
                self.mode = "idle"