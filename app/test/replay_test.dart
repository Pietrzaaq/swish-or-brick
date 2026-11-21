import 'dart:io';

import 'package:brickorswish/main.dart';
import 'package:brickorswish/replay/replay_data.dart';
import 'package:flutter_test/flutter_test.dart';

String _shared(String path) => File('../shared/$path').readAsStringSync();

void main() {
  test('real session 1: miss then six makes, all matching the verdict', () {
    const dir = 'fixtures/cases/real_session01_576p';
    final r = analyse(_shared('config/shot_logic.v1.json'), _shared('$dir/detections.json'),
        _shared('$dir/hoop.json'), _shared('$dir/expected_events.json'));
    expect(r.shots.map((s) => s.outcome), ['miss', 'make', 'make', 'make', 'make', 'make', 'make']);
    expect(r.agreed, 7);
    expect(r.verdicts, 7);
    expect(r.frames, 2292);
  });

  testWidgets('home -> replay shows the summary and shots', (tester) async {
    await tester.pumpWidget(const MainApp());
    await tester.tap(find.text('Real session 1'));
    // Not pumpAndSettle: the loading spinner animates until the result arrives.
    await tester.pump();
    await tester.pump(const Duration(seconds: 1));

    // Asset loading and the isolate run on real async, outside the fake test clock.
    for (var i = 0; i < 100 && find.text('6 / 7').evaluate().isEmpty; i++) {
      await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 50)));
      await tester.pump();
    }
    expect(find.text('6 / 7'), findsOneWidget);
    expect(find.text('Matches your verdict: 7 of 7'), findsOneWidget);
    expect(find.text('Shot 1 · Miss'), findsOneWidget);
  });
}
