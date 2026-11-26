import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:shot_logic/shot_logic.dart';
import 'package:test/test.dart';

final configUri = Directory.current.uri.resolve('../../../shared/config/shot_logic.v1.json');

Map<String, dynamic> rawConfig() =>
    jsonDecode(File.fromUri(configUri).readAsStringSync()) as Map<String, dynamic>;

void main() {
  group('config', () {
    test('loads shared config', () {
      expect(ShotLogicConfig.fromJson(rawConfig()).version, 1);
    });
    test('rejects unknown and missing keys', () {
      expect(() => ShotLogicConfig.fromJson({...rawConfig(), 'typo_key': 1}),
          throwsFormatException);
      expect(() => ShotLogicConfig.fromJson(rawConfig()..remove('gate_chi2')),
          throwsFormatException);
    });
  });

  group('geometry', () {
    test('axis aligned', () {
      final hoop = HoopFrame([900, 400], [1020, 400], [960, 415]);
      expect(hoop.radiusPx, 60.0);
      expect(hoop.toHoop(1020, 400), (1.0, 0.0));
      expect(hoop.toHoop(960, 340), (0.0, -1.0));
      expect(hoop.ellipseB, closeTo(0.25, 1e-12));
    });
    test('tilted rim does not tilt the axes', () {
      // the clicked rim is tilted (perspective / phone roll), but "down" stays the image vertical
      final hoop = HoopFrame([100, 200], [180, 260], [130, 245]);
      expect(hoop.radiusPx, 50.0);
      final (x, y) = hoop.toHoop(140, 330); // straight below the rim centre (140, 230)
      expect(x, closeTo(0.0, 1e-12));
      expect(y, closeTo(2.0, 1e-12));
      for (final (px, py) in [(0.0, 0.0), (512.5, -33.0), (140.0, 230.0)]) {
        final (hx, hy) = hoop.toHoop(px, py);
        final (bx, by) = hoop.toPixels(hx, hy);
        expect(bx, closeTo(px, 1e-9));
        expect(by, closeTo(py, 1e-9));
      }
    });
    test('measured down sets the axes', () {
      final a = 10.0 * math.pi / 180.0;
      final down = (-math.sin(a), math.cos(a));
      final hoop = HoopFrame([900, 400], [1020, 400], [960, 415], down);
      final (x, y) = hoop.toHoop(960 + 120 * down.$1, 400 + 120 * down.$2);
      expect(x, closeTo(0.0, 1e-12));
      expect(y, closeTo(2.0, 1e-12));
      expect(hoop.ellipseB, closeTo(0.25, 1e-12)); // the drawn rim does not depend on gravity
    });
    test('degenerate calibration', () {
      expect(() => HoopFrame([1, 1], [1, 1], [1, 2]), throwsArgumentError);
    });
  });

  group('tracker', () {
    test('coasts then drops', () {
      final cfg = ShotLogicConfig.fromJson(rawConfig());
      final tracker = BallTracker(cfg);
      tracker.step(0.0, [const Candidate(0.0, -5.0, 0.9)]);
      tracker.step(1 / 30, [const Candidate(0.0, -4.9, 0.9)]);
      for (var k = 0; k < cfg.maxCoastFrames; k++) {
        expect(tracker.step((2 + k) / 30, [])!.observed, isFalse);
      }
      expect(tracker.step((2 + cfg.maxCoastFrames) / 30, []), isNull);
    });
  });

  group('state machine', () {
    // Mirrors StateMachineTest in ml/tests/test_logic_units.py.
    BallState ball(double t, double x, double y, {double vy = 10.0}) => BallState(
        t: t, estX: x, estY: y, vx: 0.0, vy: vy, observed: true, trackId: 1, hits: 9, seenX: x, seenY: y);
    ShotStateMachine approach() {
      final sm = ShotStateMachine(ShotLogicConfig.fromJson(rawConfig()));
      sm.step(0.0, ball(0.0, 0.0, -5.0));
      expect(sm.state, ShotPhase.approach);
      return sm;
    }

    test('rolling on the rim edge then down the net is a make', () {
      final sm = approach();
      sm.step(0.1, ball(0.1, -0.32, -0.16));
      sm.step(0.3, ball(0.3, -1.0, 0.52));
      expect(sm.state, ShotPhase.pendingMake);
      expect(sm.step(0.33, ball(0.33, -1.19, 0.74)), isNull);
      final ev = sm.step(0.6, ball(0.6, -0.28, 2.16))!;
      expect((ev.outcome, ev.reason), ('make', 'through_rim'));
    });
    test('edge crossing then out of the net is a make', () {
      final sm = approach();
      sm.step(0.1, ball(0.1, -0.99, -1.45));
      final ev = sm.step(0.3, ball(0.3, 0.0, 2.31))!;
      expect((ev.outcome, ev.reason), ('make', 'through_net'));
    });
    test('edge crossing then beside the net is a miss', () {
      final sm = approach();
      sm.step(0.1, ball(0.1, 1.0, -0.25));
      sm.step(0.13, ball(0.13, 1.2, 0.5));
      expect(sm.state, ShotPhase.rattle);
      final ev = sm.step(0.2, ball(0.2, 1.45, 1.4))!;
      expect((ev.outcome, ev.reason), ('miss', 'rim_out'));
    });
  });

  group('static spots', () {
    // Mirrors StaticSpotTest in test_logic_units.py: the rim's bracket is detected as a ball.
    const hoopJson = {
      'image_width': 1920, 'image_height': 1080, //
      'left': [900, 400], 'right': [1020, 400], 'front': [960, 415],
    };
    Map<String, dynamic> det(double x, double y) =>
        {'cls': 'ball', 'conf': 0.3, 'x': 960 + 60 * x, 'y': 400 + 60 * y, 'w': 40.0, 'h': 40.0};
    Map<String, dynamic> session(int bracketSeconds) {
      final frames = <Map<String, dynamic>>[];
      for (var i = 0; i < 30 * 30; i++) {
        final t = i / 30.0;
        final dets = <Map<String, dynamic>>[
          if (t.floor() < bracketSeconds && i % 30 < 3) det(1.22, 0.48),
          if (t >= 10.0 && t < 10.5) det(-3.0 + 6.0 * (t - 10.0), -2.0),
        ];
        frames.add({'i': i, 't': t, 'dets': dets});
      }
      return {'fps': 30.0, 'frames': frames};
    }

    test('bracket is ignored and the ball is not', () {
      final cfg = ShotLogicConfig.fromJson(rawConfig());
      final hf = HoopFrame.fromJson(hoopJson);
      final static = staticSpots(session(cfg.staticMinSeconds), hf, cfg);
      expect(static, isNotEmpty);
      expect(ballCandidates([det(1.25, 0.45)], hf, cfg, static), isEmpty);
      expect(ballCandidates([det(1.22, -1.5)], hf, cfg, static), hasLength(1));
      expect(ballCandidates([det(-2.0, -2.0)], hf, cfg, static), hasLength(1));
    });
    test('a spot seen in few seconds is not static', () {
      final cfg = ShotLogicConfig.fromJson(rawConfig());
      expect(staticSpots(session(cfg.staticMinSeconds - 1), HoopFrame.fromJson(hoopJson), cfg), isEmpty);
    });
    test('balls dropping through the rim centre are not static', () {
      final cfg = ShotLogicConfig.fromJson(rawConfig());
      final frames = <Map<String, dynamic>>[
        for (var i = 0; i < 30 * 30; i++)
          {'i': i, 't': i / 30.0, 'dets': [if (i % 30 < 6) det(0.0, -2.0 + 0.7 * (i % 30))]},
      ];
      expect(staticSpots({'fps': 30.0, 'frames': frames}, HoopFrame.fromJson(hoopJson), cfg), isEmpty);
    });
  });
}
