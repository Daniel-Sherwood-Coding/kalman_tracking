import numpy as np

class KalmanFilter2D:
    """Simple 2D constant-velocity Kalman filter."""

    def __init__(
        self,
        initial_position,
        initial_velocity=(0.0, 0.0),
        dt=1.0,
        process_variance=1.0,
        measurement_variance=5.0,
    ):
        self.dt = float(dt)
        px, py = float(initial_position[0]), float(initial_position[1])
        vx, vy = float(initial_velocity[0]), float(initial_velocity[1])

        self.x = np.array([px, py, vx, vy], dtype=float)

        self.P = np.diag([25.0, 25.0, 100.0, 100.0])

        self.F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)

        q = float(process_variance)
        self.Q = np.array(
            [
                [q, 0.0, 0.0, 0.0],
                [0.0, q, 0.0, 0.0],
                [0.0, 0.0, q * 3.0, 0.0],
                [0.0, 0.0, 0.0, q * 3.0],
            ],
            dtype=float,
        )

        r = float(measurement_variance)
        self.R = np.array([[r, 0.0], [0.0, r]], dtype=float)

    def predict(self, dt=None):
        if dt is not None:
            self.dt = float(dt)
            self.F[0, 2] = self.dt
            self.F[1, 3] = self.dt

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, measurement):
        z = np.array([float(measurement[0]), float(measurement[1])], dtype=float)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x.copy()

    def get_position(self):
        return self.x[:2].copy()