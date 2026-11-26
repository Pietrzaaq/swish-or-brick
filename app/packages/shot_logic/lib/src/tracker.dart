import 'dart:math' as math;

import 'config.dart';
import 'kalman.dart';

class Candidate {
  final double x; // rim units
  final double y;
  final double conf;
  const Candidate(this.x, this.y, this.conf);
}

/// est*: the Kalman estimate, smooth but it assumes free fall, so it sinks below a ball resting on the rim.
/// seen*: the detection itself (the prediction while coasting). Rim decisions use seen*.
class BallState {
  final double t;
  final double estX;
  final double estY;
  final double vx;
  final double vy;
  final bool observed; // false while coasting on the prediction
  final int trackId;
  final int hits;
  final double seenX;
  final double seenY;
  const BallState({
    required this.t,
    required this.estX,
    required this.estY,
    required this.vx,
    required this.vy,
    required this.observed,
    required this.trackId,
    required this.hits,
    required this.seenX,
    required this.seenY,
  });
}

/// Single-ball tracker: picks the detection nearest to where the Kalman filter expects the ball, coasts through
/// short gaps (net, rim) on the prediction, and restarts when the prediction has run away.
class BallTracker {
  final ShotLogicConfig cfg;
  AxisKalman? _kx;
  AxisKalman? _ky;
  double _lastT = 0.0;
  int _misses = 0;
  int _hits = 0;
  int _trackId = 0;

  BallTracker(this.cfg);

  /// Any candidate can continue the track; only one >= [startMinConf] (default ballMinConf) starts one.
  BallState? step(double t, List<Candidate> candidates, [double? startMinConf]) {
    final minConf = startMinConf ?? cfg.ballMinConf;
    final kx = _kx, ky = _ky;
    if (kx != null && ky != null) {
      final dt = t - _lastT;
      _lastT = t;
      kx.predict(dt, 0.0, cfg.kalmanAccelNoise);
      ky.predict(dt, cfg.gravityRimRPerS2, cfg.kalmanAccelNoise);
      final match = _match(kx, ky, candidates);
      if (match != null) {
        final (cand, manoeuvre) = match;
        if (manoeuvre) {
          kx.inflateVelocity(cfg.kalmanInitVelVar);
          ky.inflateVelocity(cfg.kalmanInitVelVar);
        }
        kx.update(cand.x, cfg.kalmanMeasVar);
        ky.update(cand.y, cfg.kalmanMeasVar);
        _misses = 0;
        _hits += 1;
        return _state(t, observed: true, meas: cand);
      }
      _misses += 1;
      final fresh = _pickStart(candidates.where((c) => c.conf >= minConf));
      // The prediction has run away from the ball (rim / net stopped it): start over on the detection.
      final runaway = fresh != null && _misses > cfg.restartAfterCoastFrames;
      if (!runaway && _misses <= cfg.maxCoastFrames) return _state(t, observed: false);
      _kx = null;
      _ky = null;
    }

    final start = _pickStart(candidates.where((c) => c.conf >= minConf));
    if (start == null) return null;
    _trackId += 1;
    _kx = AxisKalman(start.x, cfg.kalmanMeasVar, cfg.kalmanInitVelVar);
    _ky = AxisKalman(start.y, cfg.kalmanMeasVar, cfg.kalmanInitVelVar);
    _lastT = t;
    _misses = 0;
    _hits = 1;
    return _state(t, observed: true, meas: start);
  }

  /// Best candidate inside the gate; else the nearest within reacquireRadius, flagged as a bounce.
  /// The gate: distance from the prediction in standard deviations, squared (gateChi2 9.21 = 99% of real balls).
  (Candidate, bool)? _match(AxisKalman kx, AxisKalman ky, List<Candidate> candidates) {
    final sx = kx.innovationVar(cfg.kalmanMeasVar);
    final sy = ky.innovationVar(cfg.kalmanMeasVar);
    Candidate? best, near;
    var bestD2 = double.infinity, nearD = double.infinity;
    for (final c in candidates) {
      final dx = c.x - kx.pos, dy = c.y - ky.pos;
      final d2 = dx * dx / sx + dy * dy / sy;
      if (d2 < bestD2) {
        best = c;
        bestD2 = d2;
      }
      final d = math.sqrt(dx * dx + dy * dy);
      if (d < nearD) {
        near = c;
        nearD = d;
      }
    }
    if (best != null && bestD2 <= cfg.gateChi2) return (best, false);
    if (near != null && nearD <= cfg.reacquireRadius) return (near, true);
    return null;
  }

  Candidate? _pickStart(Iterable<Candidate> candidates) {
    Candidate? best;
    for (final c in candidates) {
      if (c.x.abs() > cfg.watchHalfWidth) continue;
      if (c.y < -cfg.watchTop || c.y > cfg.watchBottom) continue;
      if (best == null || c.conf > best.conf) best = c;
    }
    return best;
  }

  BallState _state(double t, {required bool observed, Candidate? meas}) => BallState(
        t: t,
        estX: _kx!.pos,
        estY: _ky!.pos,
        vx: _kx!.vel,
        vy: _ky!.vel,
        observed: observed,
        trackId: _trackId,
        hits: _hits,
        seenX: meas?.x ?? _kx!.pos,
        seenY: meas?.y ?? _ky!.pos,
      );
}
