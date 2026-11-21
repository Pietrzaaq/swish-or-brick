// Replay: run the shot logic on a recorded session bundled with the app
// (detector output + hoop taps + your verdict), without camera or detector.
import 'dart:convert';
import 'dart:isolate';

import 'package:flutter/services.dart';
import 'package:shot_logic/shot_logic.dart';

class ReplayCase {
  const ReplayCase({required this.id, required this.title, required this.subtitle});

  /// Folder under `assets/cases/` (a copy of `shared/fixtures/cases/<id>`).
  final String id;
  final String title;
  final String subtitle;
}

/// Keep in sync with `cases` in tool/sync_assets.dart.
const replayCases = [
  ReplayCase(
    id: 'real_session01_576p',
    title: 'Real session 1',
    subtitle: '76 s phone video (576p), detected by RF-DETR Nano in Colab',
  ),
];

class ReplayShot {
  const ReplayShot({
    required this.shot,
    required this.outcome,
    required this.tStart,
    required this.tResolve,
    required this.reason,
    this.truth,
  });

  final int shot;
  final String outcome; // make | miss | uncertain
  final double tStart;
  final double tResolve;
  final String reason;

  /// Your verdict for this shot, when the case has one.
  final String? truth;

  bool? get matchesTruth => truth == null ? null : truth == outcome;
}

class ReplayResult {
  const ReplayResult({required this.shots, required this.frames, required this.durationS, required this.logicTime});

  final List<ReplayShot> shots;
  final int frames;
  final double durationS;

  /// Time spent in process() only (not JSON decoding).
  final Duration logicTime;

  int count(String outcome) => shots.where((s) => s.outcome == outcome).length;
  int get verdicts => shots.where((s) => s.truth != null).length;
  int get agreed => shots.where((s) => s.matchesTruth ?? false).length;
}

Future<ReplayResult> runReplay(ReplayCase replayCase, {AssetBundle? bundle}) async {
  final b = bundle ?? rootBundle;
  final base = 'assets/cases/${replayCase.id}';
  final texts = await Future.wait([
    b.loadString('assets/shot_logic.v1.json'),
    b.loadString('$base/detections.json'),
    b.loadString('$base/hoop.json'),
    b.loadString('$base/expected_events.json'),
  ]);
  // Off the UI thread: thousands of frames of JSON + logic.
  return Isolate.run(() => analyse(texts[0], texts[1], texts[2], texts[3]));
}

/// JSON texts -> result. Pure; separate from asset loading so tests can feed shared/ files directly.
ReplayResult analyse(String config, String detections, String hoop, String expected) {
  Map<String, dynamic> obj(String s) => jsonDecode(s) as Map<String, dynamic>;
  final cfg = ShotLogicConfig.fromJson(obj(config));
  final det = obj(detections);
  final hoopJson = obj(hoop);

  final watch = Stopwatch()..start();
  final events = process(det, hoopJson, cfg);
  watch.stop();

  final truth = {
    for (final s in (obj(expected)['shots'] as List).cast<Map<String, dynamic>>())
      s['shot'] as int: s['outcome'] as String,
  };
  final frames = det['frames'] as List;
  return ReplayResult(
    shots: [
      for (final s in (events['shots'] as List).cast<Map<String, dynamic>>())
        ReplayShot(
          shot: s['shot'] as int,
          outcome: s['outcome'] as String,
          tStart: (s['t_start'] as num).toDouble(),
          tResolve: (s['t_resolve'] as num).toDouble(),
          reason: s['reason'] as String,
          truth: truth[s['shot'] as int],
        ),
    ],
    frames: frames.length,
    durationS: frames.isEmpty ? 0 : ((frames.last as Map)['t'] as num).toDouble(),
    logicTime: watch.elapsed,
  );
}
