import 'dart:math' as math;

import 'config.dart';
import 'geometry.dart';
import 'state_machine.dart';
import 'tracker.dart';

List<Candidate> ballCandidates(List<dynamic> dets, HoopFrame hoop, ShotLogicConfig cfg) {
  final out = <Candidate>[];
  for (final raw in dets) {
    final d = raw as Map<String, dynamic>;
    final conf = (d['conf'] as num).toDouble();
    if (d['cls'] != 'ball' || conf < cfg.ballTrackMinConf) continue;
    final size = hoop.sizeInRimDiameters((d['w'] as num).toDouble(), (d['h'] as num).toDouble());
    if (!(cfg.ballMinSizeRimD <= size && size <= cfg.ballMaxSizeRimD)) continue;
    final (x, y) = hoop.toHoop((d['x'] as num).toDouble(), (d['y'] as num).toDouble());
    out.add(Candidate(x, y, conf));
  }
  return out;
}

bool inWatchZone(Candidate c, ShotLogicConfig cfg) =>
    c.x.abs() <= cfg.watchHalfWidth && -cfg.watchTop <= c.y && c.y <= cfg.watchBottom;

/// Least-squares v = a + b*t + c*t^2 -> (acceleration 2c, RMS residual).
(double, double) fitAccel(List<double> ts, List<double> vs) {
  final n = ts.length;
  var t0 = 0.0;
  for (final t in ts) {
    t0 += t;
  }
  t0 /= n;
  final s = List<double>.filled(5, 0.0); // sums of t^k, k = 0..4 (t centred for conditioning)
  final r = List<double>.filled(3, 0.0); // sums of v*t^k, k = 0..2
  for (var i = 0; i < n; i++) {
    final u = ts[i] - t0;
    var p = 1.0;
    for (var k = 0; k < 5; k++) {
      s[k] += p;
      if (k < 3) r[k] += vs[i] * p;
      p *= u;
    }
  }
  final m = [
    [s[0], s[1], s[2], r[0]],
    [s[1], s[2], s[3], r[1]],
    [s[2], s[3], s[4], r[2]],
  ];
  for (var i = 0; i < 3; i++) {
    // Gauss-Jordan; the matrix is positive definite for >= 3 distinct times
    final piv = m[i][i];
    if (piv.abs() < 1e-12) return (0.0, double.infinity);
    for (var j = i; j < 4; j++) {
      m[i][j] /= piv;
    }
    for (var k = 0; k < 3; k++) {
      if (k != i) {
        final f = m[k][i];
        for (var j = i; j < 4; j++) {
          m[k][j] -= f * m[i][j];
        }
      }
    }
  }
  final a = m[0][3], b = m[1][3], c = m[2][3];
  var sq = 0.0;
  for (var i = 0; i < n; i++) {
    final u = ts[i] - t0;
    final e = vs[i] - (a + b * u + c * u * u);
    sq += e * e;
  }
  return (2.0 * c, math.sqrt(sq / n));
}

double _median(List<double> values) {
  final v = [...values]..sort();
  final n = v.length;
  return n.isOdd ? v[n ~/ 2] : (v[n ~/ 2 - 1] + v[n ~/ 2]) / 2.0;
}

double _hypot(double a, double b) => math.sqrt(a * a + b * b);

/// Gravity as seen in this video: (unit "down" vector in the image, strength in rim radii / s^2), measured from
/// the ball's own free flights, or null if there are too few clean ones. See pipeline.py estimate_gravity().
((double, double), double)? estimateGravity(
  Map<String, dynamic> detections,
  Map<String, dynamic> hoop,
  ShotLogicConfig cfg,
) {
  final frame = HoopFrame.fromJson(hoop); // image axes: zone and size filtering only
  final n = cfg.gravityFitWindow;
  final windows = <(double, double)>[];
  var run = <(double, double, double)>[];

  void flush() {
    final stride = math.max(1, n ~/ 2);
    for (var s = 0; s + n <= run.length; s += stride) {
      final w = run.sublist(s, s + n);
      final ts = [for (final p in w) p.$1];
      final (ax, ex) = fitAccel(ts, [for (final p in w) p.$2]);
      final (ay, ey) = fitAccel(ts, [for (final p in w) p.$3]);
      if (math.max(ex, ey) <= cfg.gravityFitMaxRms) windows.add((ax, ay));
    }
  }

  for (final raw in detections['frames'] as List) {
    final fr = raw as Map<String, dynamic>;
    final balls = [
      for (final c in ballCandidates(fr['dets'] as List, frame, cfg))
        if (c.conf >= cfg.gravityFitMinConf && inWatchZone(c, cfg)) c,
    ];
    if (balls.length == 1) {
      run.add(((fr['t'] as num).toDouble(), balls[0].x, balls[0].y));
    } else {
      flush();
      run = [];
    }
  }
  flush();

  final g0 = cfg.gravityRimRPerS2;
  final plausible = [
    for (final w in windows)
      if (0.3 * g0 <= _hypot(w.$1, w.$2) && _hypot(w.$1, w.$2) <= 2.0 * g0) w,
  ];
  if (plausible.length < cfg.gravityFitMinWindows) return null;
  final ax = _median([for (final w in plausible) w.$1]);
  final ay = _median([for (final w in plausible) w.$2]);
  final mag = _hypot(ax, ay);
  if (mag <= 0.0 || ay / mag < math.cos(cfg.gravityMaxTiltDeg * math.pi / 180.0)) return null;
  return ((ax / mag, ay / mag), mag);
}

/// Hoop frame aligned with this video's gravity, and gravity strength for the tracker.
(HoopFrame, double) prepare(Map<String, dynamic> detections, Map<String, dynamic> hoop, ShotLogicConfig cfg) {
  final est = estimateGravity(detections, hoop, cfg);
  if (est == null) return (HoopFrame.fromJson(hoop), cfg.gravityRimRPerS2);
  final (down, g) = est;
  return (HoopFrame.fromJson(hoop, down), g);
}

/// detections + hoop calibration + config -> {"shots": [...]}. Pure function,
/// the contract shared with ml/brickorswish/logic/pipeline.py.
Map<String, dynamic> process(
  Map<String, dynamic> detections,
  Map<String, dynamic> hoop,
  ShotLogicConfig cfg,
) {
  final (frame, gravity) = prepare(detections, hoop, cfg);
  final tracker = BallTracker(cfg, gravity);
  final machine = ShotStateMachine(cfg);
  final shots = <Map<String, dynamic>>[];
  double? lastT;
  for (final raw in detections['frames'] as List) {
    final fr = raw as Map<String, dynamic>;
    final t = (fr['t'] as num).toDouble();
    if (lastT != null && t <= lastT) {
      throw FormatException('frame times must increase strictly (frame ${fr['i']})');
    }
    lastT = t;
    // While a shot is under way a weak detection may (re)start the track; otherwise only a confident one.
    final startConf = machine.idle ? cfg.ballMinConf : cfg.ballTrackMinConf;
    final ball = tracker.step(t, ballCandidates(fr['dets'] as List, frame, cfg), startConf);
    final event = machine.step(t, ball);
    if (event != null) shots.add(event.toJson());
  }
  if (lastT != null) {
    final event = machine.finish(lastT);
    if (event != null) shots.add(event.toJson());
  }
  return {'shots': shots};
}
