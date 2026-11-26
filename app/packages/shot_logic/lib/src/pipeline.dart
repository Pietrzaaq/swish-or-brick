import 'config.dart';
import 'geometry.dart';
import 'state_machine.dart';
import 'tracker.dart';

/// ball_in_net comes from our trained detector: still the ball, seen through the net.
const ballClasses = {'ball', 'ball_in_net'};

List<Candidate> ballCandidates(List<dynamic> dets, HoopFrame hoop, ShotLogicConfig cfg,
    [Set<(int, int)> static = const {}]) {
  final out = <Candidate>[];
  for (final raw in dets) {
    final d = raw as Map<String, dynamic>;
    final conf = (d['conf'] as num).toDouble();
    if (!ballClasses.contains(d['cls']) || conf < cfg.ballTrackMinConf) continue;
    final size = hoop.sizeInRimDiameters((d['w'] as num).toDouble(), (d['h'] as num).toDouble());
    if (!(cfg.ballMinSizeRimD <= size && size <= cfg.ballMaxSizeRimD)) continue;
    final (x, y) = hoop.toHoop((d['x'] as num).toDouble(), (d['y'] as num).toDouble());
    if (static.isNotEmpty && nearStatic(x, y, static, cfg)) continue;
    out.add(Candidate(x, y, conf));
  }
  return out;
}

// floor(v + 0.5), not round(): Python rounds halves to even, Dart away from zero.
(int, int) _cell(double x, double y, ShotLogicConfig cfg) =>
    ((x / cfg.staticCell + 0.5).floor(), (y / cfg.staticCell + 0.5).floor());

bool _near((int, int) cell, Set<(int, int)> cells) {
  for (var i = -1; i <= 1; i++) {
    for (var j = -1; j <= 1; j++) {
      if (cells.contains((cell.$1 + i, cell.$2 + j))) return true;
    }
  }
  return false;
}

bool nearStatic(double x, double y, Set<(int, int)> static, ShotLogicConfig cfg) =>
    _near(_cell(x, y, cfg), static);

/// Cells where a ball candidate stayed (also there the frame before) in >= staticMinSeconds different seconds:
/// a ball-like thing that never moves, like the rim's bracket. Counting only stays keeps out the rim centre, which
/// real balls pass through on every make. Needs the whole session (a live run would build it up as it goes).
Set<(int, int)> staticSpots(Map<String, dynamic> detections, HoopFrame hoop, ShotLogicConfig cfg) {
  final seconds = <(int, int), Set<int>>{};
  var prev = <(int, int)>{};
  for (final raw in detections['frames'] as List) {
    final fr = raw as Map<String, dynamic>;
    final sec = (fr['t'] as num).toDouble().floor();
    final cells = {for (final c in ballCandidates(fr['dets'] as List, hoop, cfg)) _cell(c.x, c.y, cfg)};
    for (final cell in cells) {
      if (_near(cell, prev)) seconds.putIfAbsent(cell, () => <int>{}).add(sec);
    }
    prev = cells;
  }
  return {
    for (final e in seconds.entries)
      if (e.value.length >= cfg.staticMinSeconds) e.key,
  };
}

/// Returns {"shots": [...]} matching shared/schema/events.schema.json.
Map<String, dynamic> process(
  Map<String, dynamic> detections,
  Map<String, dynamic> hoop,
  ShotLogicConfig cfg,
) {
  final frame = HoopFrame.fromJson(hoop);
  final static = staticSpots(detections, frame, cfg);
  final tracker = BallTracker(cfg);
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
    final ball = tracker.step(t, ballCandidates(fr['dets'] as List, frame, cfg, static), startConf);
    final event = machine.step(t, ball);
    if (event != null) shots.add(event.toJson());
  }
  if (lastT != null) {
    final event = machine.finish(lastT);
    if (event != null) shots.add(event.toJson());
  }
  return {'shots': shots};
}
