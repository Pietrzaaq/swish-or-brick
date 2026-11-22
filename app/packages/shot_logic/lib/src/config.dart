/// Typed view of `shared/config/shot_logic.v<N>.json`.
class ShotLogicConfig {
  final int version;

  final double ballMinConf;
  final double ballTrackMinConf;
  final double ballMinSizeRimD;
  final double ballMaxSizeRimD;

  /// Fallback only: process() measures gravity (direction and strength) from the ball's free flights.
  final double gravityRimRPerS2;
  final int gravityFitWindow;
  final double gravityFitMinConf;
  final double gravityFitMaxRms;
  final int gravityFitMinWindows;
  final double gravityMaxTiltDeg;
  final double kalmanAccelNoise;
  final double kalmanMeasVar;
  final double kalmanInitVelVar;
  final double gateChi2;
  final double reacquireRadius;
  final int maxCoastFrames;
  final int restartAfterCoastFrames;
  final int minHits;

  final double watchHalfWidth;
  final double watchTop;
  final double watchBottom;

  final double approachHalfWidth;
  final double approachHeight;

  final double makeMaxAbsX;
  final double rimContactAbsX;
  final double nearRimHeight;
  final double netExitMaxAbsX;
  final double makeConfirmDepth;
  final double crossMaxGapS;

  final double rattleTimeoutS;
  final double lostTimeoutS;
  final double pendingMakeLostTimeoutS;
  final double debounceS;

  static const _intKeys = {
    'version', 'max_coast_frames', 'restart_after_coast_frames', 'min_hits', //
    'gravity_fit_window', 'gravity_fit_min_windows',
  };
  static const _doubleKeys = {
    'ball_min_conf', 'ball_track_min_conf', 'ball_min_size_rim_d', 'ball_max_size_rim_d', //
    'gravity_rim_r_per_s2', 'gravity_fit_min_conf', 'gravity_fit_max_rms', 'gravity_max_tilt_deg',
    'kalman_accel_noise', 'kalman_meas_var',
    'kalman_init_vel_var', 'gate_chi2', 'reacquire_radius', //
    'watch_half_width', 'watch_top', 'watch_bottom', //
    'approach_half_width', 'approach_height', //
    'make_max_abs_x', 'rim_contact_abs_x', 'near_rim_height',
    'net_exit_max_abs_x', 'make_confirm_depth', 'cross_max_gap_s', //
    'rattle_timeout_s', 'lost_timeout_s', 'pending_make_lost_timeout_s',
    'debounce_s',
  };

  ShotLogicConfig._(Map<String, dynamic> m)
      : version = (m['version'] as num).toInt(),
        ballMinConf = _d(m, 'ball_min_conf'),
        ballTrackMinConf = _d(m, 'ball_track_min_conf'),
        ballMinSizeRimD = _d(m, 'ball_min_size_rim_d'),
        ballMaxSizeRimD = _d(m, 'ball_max_size_rim_d'),
        gravityRimRPerS2 = _d(m, 'gravity_rim_r_per_s2'),
        gravityFitWindow = (m['gravity_fit_window'] as num).toInt(),
        gravityFitMinConf = _d(m, 'gravity_fit_min_conf'),
        gravityFitMaxRms = _d(m, 'gravity_fit_max_rms'),
        gravityFitMinWindows = (m['gravity_fit_min_windows'] as num).toInt(),
        gravityMaxTiltDeg = _d(m, 'gravity_max_tilt_deg'),
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
        netExitMaxAbsX = _d(m, 'net_exit_max_abs_x'),
        makeConfirmDepth = _d(m, 'make_confirm_depth'),
        crossMaxGapS = _d(m, 'cross_max_gap_s'),
        rattleTimeoutS = _d(m, 'rattle_timeout_s'),
        lostTimeoutS = _d(m, 'lost_timeout_s'),
        pendingMakeLostTimeoutS = _d(m, 'pending_make_lost_timeout_s'),
        debounceS = _d(m, 'debounce_s');

  /// Rejects missing and unknown keys, like the Python version.
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
