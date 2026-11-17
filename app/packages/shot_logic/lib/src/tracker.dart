import 'dart:math' as math;

import 'config.dart';
import 'kalman.dart';

class Candidate {
  final double x; // hoop units
  final double y;
  final double conf;
  const Candidate(this.x, this.y, this.conf);
}

class BallState {
  final double t;
  final double x;
  final double y;
  final double vx;
  final double vy;
  final bool observed; // false while coasting on prediction
  final int trackId;
  final int hits;
  const BallState({
    required this.t,
    required this.x,
    required this.y,
    required this.vx,
    required this.vy,
    required this.observed,
    required this.trackId,
    required this.hits,
  });
}

/// Single-ball tracker: gated nearest neighbour + per-axis Kalman with gravity.
class BallTracker {
  final ShotLogicConfig cfg;
  AxisKalman? _kx;
  AxisKalman? _ky;
  double _lastT = 0.0;
  int _misses = 0;
  int _hits = 0;
  int _trackId = 0;

  BallTracker(this.cfg);

  BallState? step(double t, List<Candidate> candidates) {
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
        return _state(t, observed: true);
      }
      _misses += 1;
      if (_misses <= cfg.maxCoastFrames) return _state(t, observed: false);
      _kx = null;
      _ky = null;
    }

    final start = _pickStart(candidates);
    if (start == null) return null;
    _trackId += 1;
    _kx = AxisKalman(start.x, cfg.kalmanMeasVar, cfg.kalmanInitVelVar);
    _ky = AxisKalman(start.y, cfg.kalmanMeasVar, cfg.kalmanInitVelVar);
    _lastT = t;
    _misses = 0;
    _hits = 1;
    return _state(t, observed: true);
  }

  /// Best candidate inside the chi-square gate; else the nearest one within
  /// reacquireRadius, flagged as a manoeuvre (the ball bounced).
  (Candidate, bool)? _match(AxisKalman kx, AxisKalman ky, List<Candidate> candidates) {
    final sx = kx.innovationVar(cfg.kalmanMeasVar);
    final sy = ky.innovationVar(cfg.kalmanMeasVar);
    Candidate? best, near;
    var bestD2 = double.infinity, nearD = double.infinity;
    for (final c in candidates) {
      final dx = c.x - kx.p, dy = c.y - ky.p;
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

  Candidate? _pickStart(List<Candidate> candidates) {
    Candidate? best;
    for (final c in candidates) {
      if (c.x.abs() > cfg.watchHalfWidth) continue;
      if (c.y < -cfg.watchTop || c.y > cfg.watchBottom) continue;
      if (best == null || c.conf > best.conf) best = c;
    }
    return best;
  }

  BallState _state(double t, {required bool observed}) => BallState(
        t: t,
        x: _kx!.p,
        y: _ky!.p,
        vx: _kx!.v,
        vy: _ky!.v,
        observed: observed,
        trackId: _trackId,
        hits: _hits,
      );
}
