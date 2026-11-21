import 'config.dart';
import 'tracker.dart';

/// Make/miss state machine. See ml/brickorswish/logic/state_machine.py for the
/// transition table; this file mirrors it line for line.
///
/// Only SEEN balls decide anything. A rim crossing is placed by interpolating between the last ball seen
/// above the rim plane and the first one seen below it (at most crossMaxGapS apart, across track restarts).
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
  double? _crossX; // where this shot last crossed the rim plane downward

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
    if (event == null && ball != null && ball.observed && ball.my < 0.0) _above = ball;
    return event;
  }

  /// End of stream: an unresolved attempt is uncertain.
  ShotEvent? finish(double t) =>
      state == ShotPhase.idle ? null : _resolve(t, 'uncertain', 'stream_ended');

  void _idle(double t, BallState? ball) {
    if (ball == null || t < _cooldownUntil) return;
    final inZone = ball.x.abs() <= cfg.approachHalfWidth &&
        -cfg.approachHeight <= ball.y &&
        ball.y < 0.0;
    if (inZone && ball.vy > 0.0) {
      _tStart = t;
      _tLastSeen = t;
      _above = ball.observed ? ball : null;
      _crossX = null;
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

  /// x where the ball crossed the rim plane downward, between the last ball seen above it and this one
  /// (seen at or below it), or null.
  double? _crossing(BallState ball) {
    final above = _above;
    if (above == null || ball.my < 0.0 || ball.t - above.t > cfg.crossMaxGapS) return null;
    _above = null;
    final f = (0.0 - above.my) / (ball.my - above.my);
    final x = above.mx + f * (ball.mx - above.mx);
    _crossX = x;
    return x;
  }

  ShotEvent? _approach(double t, BallState ball) {
    final xCross = _crossing(ball);
    if (xCross != null) {
      if (xCross.abs() <= cfg.makeMaxAbsX) {
        _enter(ShotPhase.pendingMake, t);
      } else if (xCross.abs() <= cfg.rimContactAbsX) {
        _enter(ShotPhase.rattle, t);
      } else {
        return _resolve(t, 'miss', 'wide');
      }
      return null;
    }
    if (ball.vy < 0.0 && ball.my > -cfg.nearRimHeight && ball.mx.abs() <= cfg.rimContactAbsX) {
      _enter(ShotPhase.rattle, t);
      return null;
    }
    if (ball.mx.abs() > cfg.approachHalfWidth) return _resolve(t, 'miss', 'left_zone');
    return null;
  }

  ShotEvent? _pendingMake(double t, BallState ball) {
    if (ball.vy < 0.0 && ball.my < 0.0) {
      _enter(ShotPhase.rattle, t);
      return null;
    }
    if (ball.my > 0.0 && ball.mx.abs() > cfg.netExitMaxAbsX) {
      return _resolve(t, 'miss', 'rolled_out');
    }
    if (ball.my >= cfg.makeConfirmDepth) return _resolve(t, 'make', 'through_rim');
    return null;
  }

  ShotEvent? _rattle(double t, BallState ball) {
    final xCross = _crossing(ball);
    if (xCross != null) {
      if (xCross.abs() <= cfg.makeMaxAbsX) {
        _enter(ShotPhase.pendingMake, t);
        return null;
      }
      return _resolve(t, 'miss', 'rim_out');
    }
    if (ball.mx.abs() > cfg.approachHalfWidth) return _resolve(t, 'miss', 'rim_out');
    final crossX = _crossX;
    if (ball.my >= cfg.makeConfirmDepth && crossX != null && crossX.abs() > cfg.makeMaxAbsX) {
      return _resolve(t, 'miss', 'rim_out'); // went down at the edge and never came back through
    }
    return null;
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
    _crossX = null;
    _cooldownUntil = t + cfg.debounceS;
    return event;
  }
}
