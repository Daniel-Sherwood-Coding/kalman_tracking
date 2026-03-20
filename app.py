import altair as alt
import numpy as np  

from model import (KalmanTrack, BasicScenario)
from agents import TargetAgent, SearcherAgent, PredictedTargetAgent, ReserveSearcherAgent

from mesa.mesa_logging import INFO, log_to_stderr
from mesa.visualization import (
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle

log_to_stderr(INFO)


def agent_portrayal(agent):

    if isinstance(agent, TargetAgent):
        return AgentPortrayalStyle(color="red", size=40)
    elif isinstance(agent, SearcherAgent):
        return AgentPortrayalStyle(color="blue", size=40)
    elif isinstance(agent, ReserveSearcherAgent):
        return AgentPortrayalStyle(color="purple", size=50)
    elif isinstance(agent, PredictedTargetAgent):
        # Represent uncertainty as marker size; add semi-transparency.
        uncertainty = getattr(agent, "uncertainty", 0.0)
        # map uncertainty to [30, 250] pixel point size
        size = np.clip(20 + uncertainty * 4.0, 30, 250)
        return AgentPortrayalStyle(color="rgba(0,150,0,0.3)", size=size)
    else:
        return AgentPortrayalStyle(color="grey", size=20)


model_params = {
    "rng": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "n_searchers": {
        "type": "SliderInt",
        "value": 1,
        "label": "Number of searchers:",
        "min": 1,
        "max": 5,
        "step": 1,
    },
    "n_targets": {
        "type": "SliderInt",
        "value": 1,
        "label": "Number of target:",
        "min": 1,
        "max": 5,
        "step": 1,
    },
    "width": 1000,
    "height": 1000,
}


def post_process(chart):
    """Post-process the Altair chart to add a colorbar legend."""
    chart = chart.encode(
        color=alt.Color(
            "color:N",
            scale=alt.Scale(scheme="viridis", domain=[0, 10]),
            legend=alt.Legend(
                title="Wealth",
                orient="right",
                type="gradient",
                gradientLength=200,
            ),
        ),
    )
    return chart


model = KalmanTrack()

# The SpaceRenderer is responsible for drawing the model's space and agents.
# It builds the visualization in layers, first drawing the grid structure,
# and then drawing the agents on top. It uses a specified backend
# (like "altair" or "matplotlib") for creating the plots.

renderer = (
    SpaceRenderer(model, backend="altair")
    .setup_structure(  # To customize the grid appearance.
        grid_color="black", grid_dash=[6, 2], grid_opacity=0.3
    )
    .setup_agents(agent_portrayal, cmap="viridis", vmin=0, vmax=10)
)
renderer.render()

# The post_process function is used to modify the Altair chart after it has been created.
# It can be used to add legends, colorbars, or other visual elements.
renderer.post_process = post_process

# Creates a line plot component from the model's "Gini" datacollector.
# GiniPlot = make_plot_component("Gini")

# The SolaraViz page combines the model, renderer, and components into a web interface.
# To run the visualization, save this code as app.py and run `solara run app.py`
page = SolaraViz(
    model,
    renderer,
    # components=[GiniPlot],
    model_params=model_params,
    name="Kalman Model",
)
page
