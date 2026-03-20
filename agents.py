import numpy as np
from mesa.discrete_space import CellAgent
from filters import KalmanFilter2D

class TargetAgent(CellAgent):
    """ The Agent trying to get away

    Attributes:
    """

    def __init__(self, model, cell):
        """Create a new agent.

        Args:
            model (Model): The model instance that contains the agent
        """
        super().__init__(model)

        self.cell = cell
        self.velocity = 15
        self.destination = None
        self.noise_scale = 10.0  # max random deviation for snaking
        self.wobble_prob = 0.8 # chance each step to apply a noisy direction change

    def _at_destination(self):
        return self.destination is not None and tuple(self.cell.position) == tuple(self.destination.position)

    def _pick_new_destination(self):
        all_cells = list(self.model.grid.all_cells.cells)
        if not all_cells:
            self.destination = None
            return

        # choose a destination distinct from current position
        self.destination = self.model.random.choice(all_cells)
        if self._at_destination():
            self.destination = self.model.random.choice(all_cells)

    def move(self):
        """Move the agent towards the chosen destination cell over multiple steps."""
        # pick destination if needed
        if self.destination is None or self._at_destination():
            self._pick_new_destination()

        if self.destination is None:
            return

        # if we're already at destination, we will pick a new one next step
        if self._at_destination():
            return

        # pick neighborhood cell that reduces distance to a noisy/perturbed destination
        neighbors = self.cell.get_neighborhood(radius=self.velocity)
        if len(neighbors) == 0:
            return

        target_pos = np.asarray(self.destination.position, dtype=float)

        # Add temporal wobble to avoid strictly straight-line motion.
        noisy_target = target_pos
        if self.model.random.random() < self.wobble_prob:
            # Mesa's RNG supports scalar uniform; use numpy for vector noise.
            wobble = np.random.uniform(-1.0, 1.0, 2) * self.noise_scale
            noisy_target = np.clip(
                target_pos + wobble,
                [0.0, 0.0],
                [float(self.model.grid.height - 1), float(self.model.grid.width - 1)],
            )

        candidates = sorted(
            neighbors.cells,
            key=lambda c: np.linalg.norm(np.asarray(c.position, dtype=float) - noisy_target),
        )

        # Pick from top few best options to keep motion snake-like and non-deterministic.
        top_k = min(3, len(candidates))
        self.cell = self.model.random.choice(candidates[:top_k])

    def step(self):
        """Execute one step for the agent: move toward a goal, then re-target upon arrival."""
        self.move()


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

        if nearest_dist <= self.detection_radius and self.random.random() < self.senseProbability:
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
                    self.model.reserve_agent.task_search(self.kf.get_position())

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
