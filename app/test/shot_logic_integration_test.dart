// Proves the app can call the shared shot logic: runs it on one fixture case.
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:shot_logic/shot_logic.dart';

Map<String, dynamic> _json(String repoRelative) =>
    jsonDecode(File('../$repoRelative').readAsStringSync()) as Map<String, dynamic>;

void main() {
  test('app sees a swish as a make', () {
    final cfg = ShotLogicConfig.fromJson(_json('shared/config/shot_logic.v1.json'));
    const dir = 'shared/fixtures/cases/synthetic_swish';
    final result = process(_json('$dir/detections.json'), _json('$dir/hoop.json'), cfg);
    final shots = result['shots'] as List;
    expect(shots, hasLength(1));
    expect((shots.single as Map)['outcome'], 'make');
  });
}
