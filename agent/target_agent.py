import numpy as np
from mesa.discrete_space import CellAgent


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