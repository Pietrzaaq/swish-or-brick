// Mirrors ml/tests/test_fixtures.py; golden_events.json is the Python output (tools/update_golden).
import 'dart:convert';
import 'dart:io';

import 'package:shot_logic/shot_logic.dart';
import 'package:test/test.dart';

const frameTol = 1.0 / 30.0 + 1e-9;

final repoRoot = Directory.current.uri.resolve('../../../');

Map<String, dynamic> loadJson(Uri uri) =>
    jsonDecode(File.fromUri(uri).readAsStringSync()) as Map<String, dynamic>;

void main() {
  final cfg = ShotLogicConfig.fromJson(
      loadJson(repoRoot.resolve('shared/config/shot_logic.v1.json')));
  final casesDir = Directory.fromUri(repoRoot.resolve('shared/fixtures/cases/'));
  final cases = casesDir.listSync().whereType<Directory>().toList()
    ..sort((a, b) => a.path.compareTo(b.path));

  test('fixture cases exist', () => expect(cases, isNotEmpty));

  for (final dir in cases) {
    final name = dir.uri.pathSegments.where((s) => s.isNotEmpty).last;
    group(name, () {
      late List<dynamic> got;
      setUpAll(() {
        got = process(
          loadJson(dir.uri.resolve('detections.json')),
          loadJson(dir.uri.resolve('hoop.json')),
          cfg,
        )['shots'] as List<dynamic>;
      });

      test('outcomes match ground truth', () {
        final expected = loadJson(dir.uri.resolve('expected_events.json'))['shots'] as List;
        expect(got.length, expected.length);
        // known_wrong shots are wrong for a known reason; golden_events.json still pins their output.
        final keep = [for (var i = 0; i < expected.length; i++) if (!(expected[i] as Map).containsKey('known_wrong')) i];
        expect([for (final i in keep) got[i]['outcome']], [for (final i in keep) expected[i]['outcome']]);
      });

      test('matches Python reference within one frame', () {
        final golden = loadJson(dir.uri.resolve('golden_events.json'))['shots'] as List;
        expect(got.length, golden.length);
        for (var i = 0; i < got.length; i++) {
          final g = got[i] as Map, r = golden[i] as Map;
          expect(g['outcome'], r['outcome']);
          expect(g['reason'], r['reason']);
          expect(g['t_start'] as double, closeTo((r['t_start'] as num).toDouble(), frameTol));
          expect(g['t_resolve'] as double, closeTo((r['t_resolve'] as num).toDouble(), frameTol));
        }
      });
    });
  }
}
