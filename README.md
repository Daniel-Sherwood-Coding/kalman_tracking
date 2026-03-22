# Kalman filter based coordinated tracking
Target tracking is largely a solved problem (https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python). However, as an excercise, this codebase aims to develop a simple, abstract & lightweight representation of a coordinated tracking process.

# Quick Start
```bash
solara run app.py
```

# Explanation
A target indefinitely moves to randomly selected points. When the target moves into the sensor radius of the searcher, it triggers a detection with probability P. 

The searcher updates it's kalman filter prediction of the targets location, and then moves towards it. Subsequent detections update this prediction and therefore the searcher's destination.

When a given amount of time has passed without a detection of the target, the searcher agent calls upon the reserve_searcher, which moves to the last known detection of the target and conducts a spiral search from that point.