/// One-axis constant-acceleration Kalman filter, state (position, velocity).
/// Mirrors ml/brickorswish/logic/kalman.py line for line.
class AxisKalman {
  double p;
  double v = 0.0;
  double p00;
  double p01 = 0.0;
  double p11;

  AxisKalman(this.p, double measVar, double initVelVar)
      : p00 = measVar,
        p11 = initVelVar;

  void predict(double dt, double accel, double accelNoise) {
    p = p + v * dt + 0.5 * accel * dt * dt;
    v = v + accel * dt;
    final dt2 = dt * dt;
    final n00 = p00 + 2.0 * dt * p01 + dt2 * p11;
    final n01 = p01 + dt * p11;
    final n11 = p11;
    p00 = n00 + accelNoise * dt2 * dt2 / 4.0;
    p01 = n01 + accelNoise * dt2 * dt / 2.0;
    p11 = n11 + accelNoise * dt2;
  }

  double innovationVar(double measVar) => p00 + measVar;

  void update(double z, double measVar) {
    final s = p00 + measVar;
    final k0 = p00 / s;
    final k1 = p01 / s;
    final resid = z - p;
    p += k0 * resid;
    v += k1 * resid;
    final o00 = p00, o01 = p01, o11 = p11;
    p00 = (1.0 - k0) * o00;
    p01 = (1.0 - k0) * o01;
    p11 = o11 - k1 * o01;
  }

  /// Called on a manoeuvre (rim/backboard bounce): forget the old velocity.
  void inflateVelocity(double extraVar) {
    p11 += extraVar;
  }
}
