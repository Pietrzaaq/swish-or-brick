/// One-axis Kalman filter with a known acceleration (gravity): position + velocity, and how unsure it is of each.
///
/// predict() moves the ball forward by its velocity and gravity and grows the uncertainty; update() pulls it
/// towards a detection, more strongly the more unsure it was. x and y are filtered independently to keep every
/// matrix 2x2.
class AxisKalman {
  double pos;
  double vel = 0.0;

  // Uncertainty (covariance): of the position, between position and velocity, of the velocity.
  double posVar;
  double cov = 0.0;
  double velVar;

  AxisKalman(this.pos, double measVar, double initVelVar)
      : posVar = measVar,
        velVar = initVelVar;

  void predict(double dt, double accel, double accelNoise) {
    pos = pos + vel * dt + 0.5 * accel * dt * dt;
    vel = vel + accel * dt;
    final dt2 = dt * dt;
    final newPosVar = posVar + 2.0 * dt * cov + dt2 * velVar;
    final newCov = cov + dt * velVar;
    // accelNoise: how far the real ball may stray from pure free fall (spin, air, perspective).
    posVar = newPosVar + accelNoise * dt2 * dt2 / 4.0;
    cov = newCov + accelNoise * dt2 * dt / 2.0;
    velVar = velVar + accelNoise * dt2;
  }

  /// Expected spread of the next detection around the prediction.
  double innovationVar(double measVar) => posVar + measVar;

  void update(double z, double measVar) {
    final s = posVar + measVar;
    final gainPos = posVar / s;
    final gainVel = cov / s;
    final resid = z - pos;
    pos += gainPos * resid;
    vel += gainVel * resid;
    final oPosVar = posVar, oCov = cov, oVelVar = velVar;
    posVar = (1.0 - gainPos) * oPosVar;
    cov = (1.0 - gainPos) * oCov;
    velVar = oVelVar - gainVel * oCov;
  }

  /// On a rim/backboard bounce: forget the old velocity.
  void inflateVelocity(double extraVar) {
    velVar += extraVar;
  }
}
