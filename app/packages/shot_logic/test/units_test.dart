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

  group('gravity', () {
    const hoop = {
      'image_width': 1920, 'image_height': 1080, //
      'left': [900, 400], 'right': [1020, 400], 'front': [960, 415],
    };
    Map<String, dynamic> flights((double, double) down, double g, {double conf = 0.9}) {
      const r = 60.0, fps = 30.0;
      final frames = <Map<String, dynamic>>[];
      var t = 0.0;
      for (var k = 0; k < 4; k++) {
        final x0 = -4.0 + k, y0 = -9.0, vx = 4.0, vy = -6.0 + k;
        for (var i = 0; i < 20; i++) {
          final s = i / fps;
          final x = x0 + vx * s + 0.5 * g * down.$1 * s * s;
          final y = y0 + vy * s + 0.5 * g * down.$2 * s * s;
          frames.add({
            'i': frames.length, 't': t,
            'dets': [{'cls': 'ball', 'conf': conf, 'x': 960 + x * r, 'y': 400 + y * r, 'w': 63.0, 'h': 63.0}],
          });
          t += 1.0 / fps;
        }
        frames.add({'i': frames.length, 't': t, 'dets': []});
        t += 1.0 / fps;
      }
      return {'fps': fps, 'frames': frames};
    }

    test('fit is exact on a parabola', () {
      final ts = [for (var i = 0; i < 10; i++) i / 30];
      final (acc, rms) = fitAccel(ts, [for (final t in ts) 3.0 - 2.0 * t + 0.5 * 28.0 * t * t]);
      expect(acc, closeTo(28.0, 1e-6));
      expect(rms, lessThan(1e-9));
    });
    test('measures the apparent acceleration from flights', () {
      final cfg = ShotLogicConfig.fromJson(rawConfig());
      final a = 8.0 * math.pi / 180.0;
      final down = (-math.sin(a), math.cos(a));
      final (ax, ay) = estimateGravity(flights(down, 30.0), hoop, cfg)!;
      expect(ax, closeTo(30.0 * down.$1, 1e-4));
      expect(ay, closeTo(30.0 * down.$2, 1e-4));
      final (frame, accel) = prepare(flights(down, 30.0), hoop, cfg);
      expect(frame.down, (0.0, 1.0)); // axes stay on the true vertical
      expect(accel.$1, closeTo(ax, 1e-12));
    });
    test('falls back without clean flights', () {
      final cfg = ShotLogicConfig.fromJson(rawConfig());
      final weak = flights((0.0, 1.0), 30.0, conf: cfg.gravityFitMinConf / 2);
      expect(estimateGravity(weak, hoop, cfg), isNull);
      final (frame, accel) = prepare(weak, hoop, cfg);
      expect(frame.down, (0.0, 1.0));
      expect(accel, (0.0, cfg.gravityRimRPerS2));
      expect(estimateGravity(flights((1.0, 0.0), 30.0), hoop, cfg), isNull);
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
}
