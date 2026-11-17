# shot_logic

Pure Dart port of `ml/brickorswish/logic` (no Flutter, no runtime dependencies).
Must reproduce `shared/fixtures/cases/*/golden_events.json` (Python reference output) within one frame.

```sh
dart pub get
dart test -p vm
```

`pubspec.lock` is committed on purpose: it pins every dev dependency by version and sha256.
All dev dependencies come from `test` (dart.dev); every transitive package is from a verified
Google publisher except `node_preamble`, a JS string constant used only by `dart test -p node`.
