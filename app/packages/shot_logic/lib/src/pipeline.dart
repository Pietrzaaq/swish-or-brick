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

/// detections + hoop calibration + config -> {"shots": [...]}. Pure function,
/// the contract shared with ml/brickorswish/logic/pipeline.py.
Map<String, dynamic> process(
  Map<String, dynamic> detections,
  Map<String, dynamic> hoop,
  ShotLogicConfig cfg,
) {
  final frame = HoopFrame.fromJson(hoop);
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
