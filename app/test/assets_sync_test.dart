// The app bundles copies of shared/ files; they must match the originals.
// Fix a failure with: dart run tool/sync_assets.dart
import 'dart:io';

import 'package:brickorswish/replay/replay_data.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('every bundled asset equals its shared/ original', () {
    final pairs = {
      'assets/shot_logic.v1.json': '../shared/config/shot_logic.v1.json',
      for (final c in replayCases)
        for (final f in ['detections.json', 'hoop.json', 'expected_events.json'])
          'assets/cases/${c.id}/$f': '../shared/fixtures/cases/${c.id}/$f',
    };
    for (final MapEntry(key: asset, value: original) in pairs.entries) {
      expect(File(asset).existsSync(), isTrue, reason: '$asset missing: run dart run tool/sync_assets.dart');
      expect(File(asset).readAsBytesSync(), File(original).readAsBytesSync(),
          reason: '$asset differs from $original: run dart run tool/sync_assets.dart');
    }
  });
}
