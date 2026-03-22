import numpy as np
from mesa.discrete_space import CellAgent
from filters import KalmanFilter2D
from .predicted_target_agent import PredictedTargetAgent
from .target_agent import TargetAgent


class SearcherAgent(CellAgent):
    """The Agent Searching for targets with 2D Kalman tracking."""

    def __init__(self, model, cell):
        """Create a new agent.

        Args:
            model (Model): The model instance that contains the agent
        """
        super().__init__(model)
        self.cell = cell
        self.velocity = 10
        self.detection_radius = 700
        self.track_target = None
        self.last_measurement = None
        self.senseProbability = 0.5

        self.missed_detections = 0
        self.no_detection_threshold = 60

        start_pos = np.asarray(self.cell.position, dtype=float)
        self.kf = KalmanFilter2D(start_pos, initial_velocity=(0.0, 0.0), dt=1.0)

        # Predicted position marker agent (green dot)
        predicted_cell = self._find_predicted_cell()
        self.predicted_agent = PredictedTargetAgent(self.model, predicted_cell, self)

    def get_position(self):
        return np.asarray(self.cell.position, dtype=float)

    def distance_to(self, position):
        return np.linalg.norm(self.get_position() - np.asarray(position, dtype=float))

    def sense(self):
        """See the nearest target in sensor range and update the filter."""
        targets = [a for a in self.model.agents if a is not self and isinstance(a, TargetAgent)]

        if not targets:
            self.track_target = None
            return

        distances = [(t, self.distance_to(t.cell.position)) for t in targets]
        nearest, nearest_dist = min(distances, key=lambda d: d[1])

        if nearest_dist <= self.detection_radius and self.model.random.random() < self.senseProbability:
            measurement = np.asarray(nearest.cell.position, dtype=float)
            self.kf.update(measurement)
            self.track_target = nearest
            self.last_measurement = measurement
            self.missed_detections = 0
            # Share the detection with all searchers and reserve agent
            self.model.broadcast_target_detection(measurement)
        else:
            self.track_target = None
            self.missed_detections += 1
            if self.missed_detections >= self.no_detection_threshold:
                self.missed_detections = 0
                if hasattr(self.model, "reserve_agent") and self.model.reserve_agent is not None:
                    # Use last known target position if available, otherwise fall back to filter prediction.
                    search_position = (
                        self.model.last_known_target_position
                        if self.model.last_known_target_position is not None
                        else self.kf.get_position()
                    )
                    self.model.reserve_agent.task_search(search_position)

    def _find_predicted_cell(self):
        target_pos = self.kf.get_position()
        try:
            return self.model.grid.find_nearest_cell(target_pos)
        except ValueError:
            # clamp into bounds if outside (non-torus grid)
            clamped = np.clip(
                np.floor(target_pos).astype(int),
                [0, 0],
                [self.model.grid.height - 1, self.model.grid.width - 1],
            )
            return self.model.grid._cells[tuple(clamped)]

    def update_predicted_marker(self):
        new_predicted_cell = self._find_predicted_cell()
        # Use positional covariance as uncertainty proxy (2D trace of P).
        cov2d = self.kf.P[:2, :2]
        uncertainty = np.sqrt(np.trace(cov2d))

        if self.predicted_agent is None:
            self.predicted_agent = PredictedTargetAgent(
                self.model,
                new_predicted_cell,
                self,
                uncertainty=uncertainty,
            )
        else:
            self.predicted_agent.cell = new_predicted_cell
            self.predicted_agent.uncertainty = float(uncertainty)

    def move(self):
        """Move towards the Kalman-predicted target position (or random fallback)."""
        self.kf.predict()
        target_pos = self.kf.get_position()

        neighbors = self.cell.get_neighborhood(radius=self.velocity)
        if len(neighbors) == 0:
            return

        best = min(
            neighbors.cells,
            key=lambda c: np.linalg.norm(np.asarray(c.position, dtype=float) - target_pos),
        )

        self.cell = best

    def step(self):
        """Execute one step for the agent: sense -> move -> update prediction marker."""
        if self.model.time % 20 == 0:  # Sense every 20 steps
            self.sense()
        self.move()
        self.update_predicted_marker()