import 'config.dart';
import 'tracker.dart';

/// Make/miss state machine. See ml/brickorswish/logic/state_machine.py for the
/// transition table; this file mirrors it line for line.
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
  BallState? _prev;

  ShotStateMachine(this.cfg);

  ShotEvent? step(double t, BallState? ball) {
    if (ball != null && ball.hits < cfg.minHits) ball = null;
    var prev = _prev;
    if (prev != null && (ball == null || ball.trackId != prev.trackId)) prev = null;
    _prev = ball;
    if (ball != null && ball.observed) _tLastSeen = t;

    if (state == ShotPhase.idle) {
      _idle(t, ball);
      return null;
    }
    return _active(t, ball, prev);
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
      _enter(ShotPhase.approach, t);
    }
  }

  ShotEvent? _active(double t, BallState? ball, BallState? prev) {
    final pending = state == ShotPhase.pendingMake;
    final lostLimit = pending ? cfg.pendingMakeLostTimeoutS : cfg.lostTimeoutS;
    if (t - _tLastSeen > lostLimit) {
      return _resolve(t, 'uncertain', pending ? 'lost_in_net' : 'lost');
    }
    if (state == ShotPhase.rattle && t - _tState > cfg.rattleTimeoutS) {
      return _resolve(t, 'uncertain', 'rattle_timeout');
    }
    if (ball == null) return null;

    return switch (state) {
      ShotPhase.approach => _approach(t, ball, prev),
      ShotPhase.pendingMake => _pendingMake(t, ball),
      _ => _rattle(t, ball, prev),
    };
  }

  ShotEvent? _approach(double t, BallState ball, BallState? prev) {
    final xCross = _crossingX(prev, ball);
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
    if (ball.observed &&
        ball.vy < 0.0 &&
        ball.y > -cfg.nearRimHeight &&
        ball.x.abs() <= cfg.rimContactAbsX) {
      _enter(ShotPhase.rattle, t);
      return null;
    }
    if (ball.x.abs() > cfg.approachHalfWidth) return _resolve(t, 'miss', 'left_zone');
    return null;
  }

  ShotEvent? _pendingMake(double t, BallState ball) {
    if (!ball.observed) return null; // the net hides the ball; never decide on a prediction
    if (ball.vy < 0.0 && ball.y < 0.0) {
      _enter(ShotPhase.rattle, t);
      return null;
    }
    if (ball.y > 0.0 && ball.x.abs() > cfg.netExitMaxAbsX) {
      return _resolve(t, 'miss', 'rolled_out');
    }
    if (ball.y >= cfg.makeConfirmDepth) return _resolve(t, 'make', 'through_rim');
    return null;
  }

  ShotEvent? _rattle(double t, BallState ball, BallState? prev) {
    final xCross = _crossingX(prev, ball);
    if (xCross != null) {
      if (xCross.abs() <= cfg.makeMaxAbsX) {
        _enter(ShotPhase.pendingMake, t);
        return null;
      }
      return _resolve(t, 'miss', 'rim_out');
    }
    if (ball.observed && ball.x.abs() > cfg.approachHalfWidth) {
      return _resolve(t, 'miss', 'rim_out');
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
    _cooldownUntil = t + cfg.debounceS;
    return event;
  }
}

/// x where the ball crossed the rim plane (y=0) moving down, linearly interpolated.
double? _crossingX(BallState? prev, BallState cur) {
  if (prev == null || !(prev.y < 0.0 && 0.0 <= cur.y)) return null;
  final f = (0.0 - prev.y) / (cur.y - prev.y);
  return prev.x + f * (cur.x - prev.x);
}
