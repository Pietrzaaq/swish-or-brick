import 'dart:convert';
import 'dart:io';

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
    test('tilted roundtrip', () {
      final hoop = HoopFrame([100, 200], [180, 260], [130, 245]);
      final (x, y) = hoop.toHoop(180, 260);
      expect(x, closeTo(1.0, 1e-12));
      expect(y, closeTo(0.0, 1e-12));
      for (final (px, py) in [(0.0, 0.0), (512.5, -33.0), (140.0, 230.0)]) {
        final (hx, hy) = hoop.toHoop(px, py);
        final (bx, by) = hoop.toPixels(hx, hy);
        expect(bx, closeTo(px, 1e-9));
        expect(by, closeTo(py, 1e-9));
      }
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
}
