"""
Kalman Filter Model
=====================

A simple model of two agents, one trying to search and then track another.
"""

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.experimental.data_collection import DataRecorder, DatasetConfig
from mesa.experimental.scenarios import Scenario

from agents import SearcherAgent
from agents import TargetAgent

import numpy as np

class BasicScenario(Scenario):
    """Scenario parameters for the basic scenario."""

    n_searchers: int = 1
    n_targets: int = 1
    width: int = 1000
    height: int = 1000


class KalmanTrack(Model):
    """ A simple model of two entities, with Blue trying to track Red.

    Blue wins by getting 5 consecutive successful tracks on Red

    Red wins by getting at least 20 cells away from Blue

    Attributes:
        num_agents (int): Number of agents in the model
        grid (MultiGrid): The space in which agents move
        running (bool): Whether the model should continue running
        datacollector (DataCollector): Collects and stores model data
    """

    def __init__(self, scenario=None):
        """Initialize the model.

        Args:
            scenario: BasicScenario object containing model parameters.
        """
        if scenario is None:
            scenario = BasicScenario()

        super().__init__(scenario=scenario)

        self.num_searchers = scenario.n_searchers
        self.num_targets = scenario.n_targets
        self.grid = OrthogonalMooreGrid(
            (scenario.width, scenario.height), random=self.random
        )

        # self.recorder = DataRecorder(self)
        # (
        #     self.data_registry.track_agents(self.agents, "agent_data", "wealth").record(
        #         self.recorder
        #     )
        # )
        # (
        #     self.data_registry.track_model(self, "model_data", "gini").record(
        #         self.recorder, configuration=DatasetConfig(start_time=4, interval=2)
        #     )
        # )

        # # Set up data collection
        # self.datacollector = DataCollector(
        #     model_reporters={"Gini": "gini"},
        #     agent_reporters={"Wealth": "wealth"},
        # )
        TargetAgent.create_agents(
            self,
            self.num_targets,
            self.random.choices(self.grid.all_cells.cells, k=self.num_targets),  # TODO: Rather that a random starting cell, we should define a fixed cell (e.g. 500,500)
        )
        SearcherAgent.create_agents(
            self,
            self.num_searchers,
            self.random.choices(self.grid.all_cells.cells, k=self.num_searchers), # TODO: Rather that a random starting cell, we should define a call offset from the target starting position (e.g. neighbourhood(n=100))
        )

        # reserve searcher sits at a fixed base location and is deployed when regular searcher loses track
        base_x = min(500, self.grid.width - 1)
        base_y = min(1000, self.grid.height - 1)
        base_cell = self.grid.find_nearest_cell((base_x, base_y))
        from agents import ReserveSearcherAgent

        self.reserve_agent = ReserveSearcherAgent.create_agents(self, 1, base_cell)[0]

        self.last_known_target_position = None
        self.running = True
        # self.datacollector.collect(self)

    def step(self):
        self.agents.shuffle_do("step")  # Activate all agents in random order
        # self.datacollector.collect(self)  # Collect data

    def broadcast_target_detection(self, measurement):
        self.last_known_target_position = np.asarray(measurement, dtype=float)
        for agent in self.agents:
            if isinstance(agent, SearcherAgent):
                agent.kf.update(self.last_known_target_position)
                agent.track_target = None
                agent.last_measurement = self.last_known_target_position
                agent.missed_detections = 0

    # @property
    # def gini(self):
    #     """Calculate the Gini coefficient for the model's current wealth distribution.

    #     The Gini coefficient is a measure of inequality in distributions.
    #     - A Gini of 0 represents complete equality, where all agents have equal wealth.
    #     - A Gini of 1 represents maximal inequality, where one agent has all wealth.
    #     """
    #     agent_wealths = [agent.wealth for agent in self.agents]
    #     x = sorted(agent_wealths)
    #     n = self.num_agents
    #     # Calculate using the standard formula for Gini coefficient
    #     b = sum(xi * (n - i) for i, xi in enumerate(x)) / (n * sum(x))
    #     return 1 + (1 / n) - 2 * b
