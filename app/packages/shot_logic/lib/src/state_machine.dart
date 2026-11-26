import 'config.dart';
import 'tracker.dart';

/// Make/miss state machine; transition table and rationale in ml/brickorswish/logic/state_machine.py.
///
/// Positions are in rim units (geometry.dart): (0, 0) is the rim centre, 1 = rim radius, y grows downward, so
/// y < 0 is above the rim and |x| < 1 is inside it.
///
/// Only SEEN balls decide. A rim crossing is interpolated between the last ball seen above the rim plane and
/// the first seen below it, even across track restarts. A ball rolling around the rim does not follow the
/// straight line, so an edge crossing is decided by where the ball comes out below ([_exit]).
enum ShotPhase { idle, approach, pendingMake, rattle }

class ShotEvent {
  final int shot;
  final String outcome; // make | miss | uncertain
  final double tStart;
  final double tResolve;
  final String reason;
  const ShotEvent(this.shot, this.outcome, this.tStart, this.tResolve, this.reason);

  Map<String, dynamic> toJson() => {
        'shot': shot,
        'outcome': outcome,
        't_start': _round4(tStart),
        't_resolve': _round4(tResolve),
        'reason': reason,
      };

  static double _round4(double v) => (v * 10000).roundToDouble() / 10000;
}

class ShotStateMachine {
  final ShotLogicConfig cfg;
  ShotPhase state = ShotPhase.idle;
  int _shots = 0;
  double _tStart = 0.0;
  double _tState = 0.0;
  double _tLastSeen = 0.0;
  double _cooldownUntil = -1.0;
  BallState? _above; // last ball SEEN above the rim plane during this shot

  ShotStateMachine(this.cfg);

  bool get idle => state == ShotPhase.idle;

  ShotEvent? step(double t, BallState? ball) {
    if (state == ShotPhase.idle) {
      if (ball != null && ball.hits < cfg.minHits) ball = null; // a shot only starts on a confirmed track
      if (ball != null && ball.observed) _tLastSeen = t;
      _idle(t, ball);
      return null;
    }

    if (ball != null && ball.observed) _tLastSeen = t;
    final event = _active(t, ball);
    if (event == null && ball != null && ball.observed && ball.seenY < 0.0) _above = ball;
    return event;
  }

  /// End of stream: an unresolved attempt is uncertain.
  ShotEvent? finish(double t) =>
      state == ShotPhase.idle ? null : _resolve(t, 'uncertain', 'stream_ended');

  void _idle(double t, BallState? ball) {
    if (ball == null || t < _cooldownUntil) return;
    final inZone = ball.estX.abs() <= cfg.approachHalfWidth &&
        -cfg.approachHeight <= ball.estY &&
        ball.estY < 0.0;
    if (inZone && ball.vy > 0.0) {
      _tStart = t;
      _tLastSeen = t;
      _above = ball.observed ? ball : null;
      _enter(ShotPhase.approach, t);
    }
  }

  ShotEvent? _active(double t, BallState? ball) {
    final pending = state == ShotPhase.pendingMake;
    final lostLimit = pending ? cfg.pendingMakeLostTimeoutS : cfg.lostTimeoutS;
    if (t - _tLastSeen > lostLimit) {
      return _resolve(t, 'uncertain', pending ? 'lost_in_net' : 'lost');
    }
    if (state == ShotPhase.rattle && t - _tState > cfg.rattleTimeoutS) {
      return _resolve(t, 'uncertain', 'rattle_timeout');
    }
    if (ball == null || !ball.observed) return null; // never decide on a prediction

    return switch (state) {
      ShotPhase.approach => _approach(t, ball),
      ShotPhase.pendingMake => _pendingMake(t, ball),
      _ => _rattle(t, ball),
    };
  }

  /// x where the ball crossed the rim plane downward since the last sighting above it, or null.
  double? _crossing(BallState ball) {
    final above = _above;
    if (above == null || ball.seenY < 0.0 || ball.t - above.t > cfg.crossMaxGapS) return null;
    _above = null;
    final f = (0.0 - above.seenY) / (ball.seenY - above.seenY);
    return above.seenX + f * (ball.seenX - above.seenX);
  }

  /// A ball seen at least exitDepth below the rim plane after touching the rim: out of the net, or beside it.
  ShotEvent? _exit(double t, BallState ball) {
    if (ball.seenY < cfg.exitDepth) return null;
    if (ball.seenX.abs() <= cfg.exitMaxAbsX) return _resolve(t, 'make', 'through_net');
    return _resolve(t, 'miss', 'rim_out');
  }

  ShotEvent? _approach(double t, BallState ball) {
    final xCross = _crossing(ball);
    if (xCross != null) {
      if (xCross.abs() <= cfg.makeMaxAbsX) {
        _enter(ShotPhase.pendingMake, t);
      } else if (xCross.abs() <= cfg.rimContactAbsX) {
        _enter(ShotPhase.rattle, t);
        return _exit(t, ball); // this sighting may already be below the net
      } else {
        return _resolve(t, 'miss', 'wide');
      }
      return null;
    }
    if (ball.vy < 0.0 && ball.seenY > -cfg.nearRimHeight && ball.seenX.abs() <= cfg.rimContactAbsX) {
      _enter(ShotPhase.rattle, t);
      return null;
    }
    if (ball.seenX.abs() > cfg.approachHalfWidth) return _resolve(t, 'miss', 'left_zone');
    return null;
  }

  ShotEvent? _pendingMake(double t, BallState ball) {
    if (ball.vy < 0.0 && ball.seenY < 0.0) {
      _enter(ShotPhase.rattle, t);
      return null;
    }
    if (ball.seenY >= cfg.makeConfirmDepth) return _resolve(t, 'make', 'through_rim');
    return null;
  }

  ShotEvent? _rattle(double t, BallState ball) {
    final xCross = _crossing(ball);
    if (xCross != null && xCross.abs() <= cfg.makeMaxAbsX) {
      _enter(ShotPhase.pendingMake, t);
      return null;
    }
    if (xCross != null && xCross.abs() > cfg.rimContactAbsX) return _resolve(t, 'miss', 'rim_out');
    if (ball.seenX.abs() > cfg.approachHalfWidth) return _resolve(t, 'miss', 'rim_out');
    return _exit(t, ball);
  }

  void _enter(ShotPhase s, double t) {
    state = s;
    _tState = t;
  }

  ShotEvent _resolve(double t, String outcome, String reason) {
    _shots += 1;
    final event = ShotEvent(_shots, outcome, _tStart, t, reason);
    state = ShotPhase.idle;
    _above = null;
    _cooldownUntil = t + cfg.debounceS;
    return event;
  }
}
