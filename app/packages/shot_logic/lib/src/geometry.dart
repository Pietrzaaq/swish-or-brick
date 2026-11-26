import 'dart:math' as math;

/// Rim units, used by all the logic: origin at the rim centre, 1 unit = rim radius, y along [down], x
/// perpendicular to it (to the right). So y < 0 is above the rim, |x| < 1 is inside it, and the same numbers
/// work for any video size or distance from the hoop.
///
/// [down] is the true vertical (image vertical for an upright phone, later the gravity sensor). Not the clicked
/// rim's tilt, which is perspective and would make a straight fall drift sideways, nor the ball's apparent fall
/// direction, which perspective also bends.
class HoopFrame {
  late final double cx;
  late final double cy;
  late final double radiusPx;
  late final (double, double) down;

  /// The rim as clicked, for drawing only; [ellipseB] is the semi-minor axis in rim units.
  late final double rimAngle;

  late final double ellipseB;

  HoopFrame(List<num> left, List<num> right, List<num> front, [(double, double) down = (0.0, 1.0)]) {
    final lx = left[0].toDouble(), ly = left[1].toDouble();
    final rx = right[0].toDouble(), ry = right[1].toDouble();
    final dx = rx - lx, dy = ry - ly;
    final length = math.sqrt(dx * dx + dy * dy);
    if (length <= 0.0) {
      throw ArgumentError('left and right rim points must differ');
    }
    final (gx, gy) = down;
    final g = math.sqrt(gx * gx + gy * gy);
    if (g <= 0.0) throw ArgumentError('down must be a non-zero vector');
    cx = (lx + rx) / 2.0;
    cy = (ly + ry) / 2.0;
    radiusPx = length / 2.0;
    this.down = (gx / g, gy / g);
    rimAngle = math.atan2(dy, dx);
    final fx = front[0].toDouble() - cx, fy = front[1].toDouble() - cy;
    ellipseB = (fx * dy - fy * dx).abs() / length / radiusPx;
  }

  factory HoopFrame.fromJson(Map<String, dynamic> m, [(double, double) down = (0.0, 1.0)]) => HoopFrame(
        (m['left'] as List).cast<num>(),
        (m['right'] as List).cast<num>(),
        (m['front'] as List).cast<num>(),
        down,
      );

  (double, double) toHoop(double px, double py) {
    final tx = px - cx, ty = py - cy;
    final (gx, gy) = down;
    // x axis = down rotated 90 degrees anticlockwise: (gy, -gx)
    return ((tx * gy - ty * gx) / radiusPx, (tx * gx + ty * gy) / radiusPx);
  }

  (double, double) toPixels(double x, double y) {
    final (gx, gy) = down;
    final sx = x * radiusPx, sy = y * radiusPx;
    return (sx * gy + sy * gx + cx, -sx * gx + sy * gy + cy);
  }

  double sizeInRimDiameters(double wPx, double hPx) =>
      ((wPx + hPx) / 2.0) / (2.0 * radiusPx);
}
