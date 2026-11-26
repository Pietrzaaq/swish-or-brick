"""One-axis Kalman filter with a known acceleration (gravity): position + velocity, and how unsure it is of each.

predict() moves the ball forward by its velocity and gravity and grows the uncertainty; update() pulls it towards
a detection, more strongly the more unsure it was. x and y are filtered independently to keep every matrix 2x2.
"""


class AxisKalman:
    def __init__(self, pos: float, meas_var: float, init_vel_var: float):
        self.pos = pos
        self.vel = 0.0
        # Uncertainty (covariance): of the position, between position and velocity, of the velocity.
        self.pos_var = meas_var
        self.cov = 0.0
        self.vel_var = init_vel_var

    def predict(self, dt: float, accel: float, accel_noise: float) -> None:
        self.pos = self.pos + self.vel * dt + 0.5 * accel * dt * dt
        self.vel = self.vel + accel * dt
        dt2 = dt * dt
        pos_var = self.pos_var + 2.0 * dt * self.cov + dt2 * self.vel_var
        cov = self.cov + dt * self.vel_var
        # accel_noise: how far the real ball may stray from pure free fall (spin, air, perspective).
        self.pos_var = pos_var + accel_noise * dt2 * dt2 / 4.0
        self.cov = cov + accel_noise * dt2 * dt / 2.0
        self.vel_var = self.vel_var + accel_noise * dt2

    def innovation_var(self, meas_var: float) -> float:
        """Expected spread of the next detection around the prediction."""
        return self.pos_var + meas_var

    def update(self, z: float, meas_var: float) -> None:
        s = self.pos_var + meas_var
        gain_pos = self.pos_var / s
        gain_vel = self.cov / s
        resid = z - self.pos
        self.pos += gain_pos * resid
        self.vel += gain_vel * resid
        pos_var, cov, vel_var = self.pos_var, self.cov, self.vel_var
        self.pos_var = (1.0 - gain_pos) * pos_var
        self.cov = (1.0 - gain_pos) * cov
        self.vel_var = vel_var - gain_vel * cov

    def inflate_velocity(self, extra_var: float) -> None:
        """On a rim/backboard bounce: forget the old velocity."""
        self.vel_var += extra_var
