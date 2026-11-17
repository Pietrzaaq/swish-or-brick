"""One-axis constant-acceleration Kalman filter with state (position, velocity).

x and y are filtered independently (diagonal process noise), which keeps every
matrix 2x2 and the Dart port trivial. Gravity enters as a known control input on y.
"""


class AxisKalman:
    def __init__(self, pos: float, meas_var: float, init_vel_var: float):
        self.p = pos
        self.v = 0.0
        self.p00 = meas_var
        self.p01 = 0.0
        self.p11 = init_vel_var

    def predict(self, dt: float, accel: float, accel_noise: float) -> None:
        self.p = self.p + self.v * dt + 0.5 * accel * dt * dt
        self.v = self.v + accel * dt
        dt2 = dt * dt
        n00 = self.p00 + 2.0 * dt * self.p01 + dt2 * self.p11
        n01 = self.p01 + dt * self.p11
        n11 = self.p11
        self.p00 = n00 + accel_noise * dt2 * dt2 / 4.0
        self.p01 = n01 + accel_noise * dt2 * dt / 2.0
        self.p11 = n11 + accel_noise * dt2

    def innovation_var(self, meas_var: float) -> float:
        return self.p00 + meas_var

    def update(self, z: float, meas_var: float) -> None:
        s = self.p00 + meas_var
        k0 = self.p00 / s
        k1 = self.p01 / s
        resid = z - self.p
        self.p += k0 * resid
        self.v += k1 * resid
        p00, p01, p11 = self.p00, self.p01, self.p11
        self.p00 = (1.0 - k0) * p00
        self.p01 = (1.0 - k0) * p01
        self.p11 = p11 - k1 * p01

    def inflate_velocity(self, extra_var: float) -> None:
        """Called on a manoeuvre (rim/backboard bounce): forget the old velocity."""
        self.p11 += extra_var
