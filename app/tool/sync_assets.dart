// Copies the shared thresholds and the bundled replay cases into app/assets/.
// Flutter can only bundle files inside the app folder; test/assets_sync_test.dart
// fails when these copies drift from shared/. Run from app/:
//
//   dart run tool/sync_assets.dart
import 'dart:io';

/// Must list every ReplayCase.id in lib/replay/replay_data.dart.
const cases = ['real_session01_576p'];
const caseFiles = ['detections.json', 'hoop.json', 'expected_events.json'];

void main() {
  _copy('../shared/config/shot_logic.v1.json', 'assets/shot_logic.v1.json');
  for (final c in cases) {
    for (final f in caseFiles) {
      _copy('../shared/fixtures/cases/$c/$f', 'assets/cases/$c/$f');
    }
  }
}

void _copy(String from, String to) {
  File(to).parent.createSync(recursive: true);
  File(from).copySync(to);
  stdout.writeln('$from -> $to');
}
