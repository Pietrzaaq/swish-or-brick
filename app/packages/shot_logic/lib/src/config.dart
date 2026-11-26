/// Typed view of `shared/config/shot_logic.v<N>.json`.
///
/// Distances are in rim units (1 = rim radius, see geometry.dart), times in seconds.
class ShotLogicConfig {
  final int version;

  // --- 1. Which detections count as the ball ---
  /// A track starts only on ballMinConf; during a shot ballTrackMinConf is enough
  /// (the ball fades to 15-30% confidence where it overlaps rim, net, board).
  final double ballMinConf;
  final double ballTrackMinConf;
  final double ballMinSizeRimD; // box size, in rim diameters
  final double ballMaxSizeRimD;

  /// Ball-like things that never move (the rim's bracket): a staticCell-sized cell where a ball candidate stayed
  /// in >= staticMinSeconds different seconds is ignored, with its neighbours.
  final double staticCell;
  final int staticMinSeconds;

  // --- 2. Following the ball (tracker + Kalman) ---
  final double gravityRimRPerS2; // 9.81 m/s^2 / 0.2286 m rim radius
  final double kalmanAccelNoise;
  final double kalmanMeasVar;
  final double kalmanInitVelVar;
  final double gateChi2;
  final double reacquireRadius; // nearest detection this close continues the track after a bounce
  final int maxCoastFrames; // frames followed on the prediction alone before the track is dropped

  /// After this many frames without a sighting, a detection outside the gate restarts the track
  /// (the prediction has run away from a ball the rim or net stopped).
  final int restartAfterCoastFrames;
  final int minHits; // sightings before a track may start a shot

  /// Region where a new track may start.
  final double watchHalfWidth;
  final double watchTop;
  final double watchBottom;

  // --- 3. Make / miss decisions (state_machine.dart) ---
  /// Approach zone (above the rim).
  final double approachHalfWidth;
  final double approachHeight;
  final double makeMaxAbsX;
  final double rimContactAbsX;
  final double nearRimHeight;
  final double makeConfirmDepth;

  /// Max gap between the last sighting above the rim plane and the first below it for a crossing to count.
  final double crossMaxGapS;

  /// After an edge crossing or a rattle, the first sighting >= exitDepth below the rim plane decides: within
  /// exitMaxAbsX it came out of the net (make), else it fell beside it.
  /// exitMaxAbsX: a ball's centre inside the net is within rim radius - ball radius ~ 0.47.
  final double exitDepth;
  final double exitMaxAbsX;

  // --- 4. Timeouts ---
  final double rattleTimeoutS;
  final double lostTimeoutS;
  final double pendingMakeLostTimeoutS;
  final double debounceS; // no new shot this soon after the last one

  static const _intKeys = {
    'version', 'static_min_seconds', 'max_coast_frames', 'restart_after_coast_frames', 'min_hits', //
  };
  static const _doubleKeys = {
    'ball_min_conf', 'ball_track_min_conf', 'ball_min_size_rim_d', 'ball_max_size_rim_d', 'static_cell', //
    'gravity_rim_r_per_s2', 'kalman_accel_noise', 'kalman_meas_var', 'kalman_init_vel_var', 'gate_chi2',
    'reacquire_radius', 'watch_half_width', 'watch_top', 'watch_bottom', //
    'approach_half_width', 'approach_height', 'make_max_abs_x', 'rim_contact_abs_x', 'near_rim_height',
    'make_confirm_depth', 'cross_max_gap_s', 'exit_depth', 'exit_max_abs_x', //
    'rattle_timeout_s', 'lost_timeout_s', 'pending_make_lost_timeout_s', 'debounce_s',
  };

  ShotLogicConfig._(Map<String, dynamic> m)
      : version = (m['version'] as num).toInt(),
        ballMinConf = _d(m, 'ball_min_conf'),
        ballTrackMinConf = _d(m, 'ball_track_min_conf'),
        ballMinSizeRimD = _d(m, 'ball_min_size_rim_d'),
        ballMaxSizeRimD = _d(m, 'ball_max_size_rim_d'),
        staticCell = _d(m, 'static_cell'),
        staticMinSeconds = (m['static_min_seconds'] as num).toInt(),
        gravityRimRPerS2 = _d(m, 'gravity_rim_r_per_s2'),
        kalmanAccelNoise = _d(m, 'kalman_accel_noise'),
        kalmanMeasVar = _d(m, 'kalman_meas_var'),
        kalmanInitVelVar = _d(m, 'kalman_init_vel_var'),
        gateChi2 = _d(m, 'gate_chi2'),
        reacquireRadius = _d(m, 'reacquire_radius'),
        maxCoastFrames = (m['max_coast_frames'] as num).toInt(),
        restartAfterCoastFrames = (m['restart_after_coast_frames'] as num).toInt(),
        minHits = (m['min_hits'] as num).toInt(),
        watchHalfWidth = _d(m, 'watch_half_width'),
        watchTop = _d(m, 'watch_top'),
        watchBottom = _d(m, 'watch_bottom'),
        approachHalfWidth = _d(m, 'approach_half_width'),
        approachHeight = _d(m, 'approach_height'),
        makeMaxAbsX = _d(m, 'make_max_abs_x'),
        rimContactAbsX = _d(m, 'rim_contact_abs_x'),
        nearRimHeight = _d(m, 'near_rim_height'),
        makeConfirmDepth = _d(m, 'make_confirm_depth'),
        crossMaxGapS = _d(m, 'cross_max_gap_s'),
        exitDepth = _d(m, 'exit_depth'),
        exitMaxAbsX = _d(m, 'exit_max_abs_x'),
        rattleTimeoutS = _d(m, 'rattle_timeout_s'),
        lostTimeoutS = _d(m, 'lost_timeout_s'),
        pendingMakeLostTimeoutS = _d(m, 'pending_make_lost_timeout_s'),
        debounceS = _d(m, 'debounce_s');

  factory ShotLogicConfig.fromJson(Map<String, dynamic> m) {
    final known = {..._intKeys, ..._doubleKeys};
    final missing = known.where((k) => !m.containsKey(k)).toList()..sort();
    final unknown = m.keys.where((k) => !known.contains(k)).toList()..sort();
    if (missing.isNotEmpty || unknown.isNotEmpty) {
      throw FormatException('config mismatch: missing=$missing unknown=$unknown');
    }
    return ShotLogicConfig._(m);
  }

  static double _d(Map<String, dynamic> m, String k) => (m[k] as num).toDouble();
}
