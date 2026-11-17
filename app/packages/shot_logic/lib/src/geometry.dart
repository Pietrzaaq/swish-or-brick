import 'dart:math' as math;

/// Origin at rim centre, x from left to right rim edge, 1 unit = rim radius, y down.
class HoopFrame {
  late final double cx;
  late final double cy;
  late final double radiusPx;
  late final double _cos;
  late final double _sin;
  late final double ellipseB;

  HoopFrame(List<num> left, List<num> right, List<num> front) {
    final lx = left[0].toDouble(), ly = left[1].toDouble();
    final rx = right[0].toDouble(), ry = right[1].toDouble();
    final dx = rx - lx, dy = ry - ly;
    final length = math.sqrt(dx * dx + dy * dy);
    if (length <= 0.0) {
      throw ArgumentError('left and right rim points must differ');
    }
    cx = (lx + rx) / 2.0;
    cy = (ly + ry) / 2.0;
    radiusPx = length / 2.0;
    _cos = dx / length;
    _sin = dy / length;
    ellipseB = toHoop(front[0].toDouble(), front[1].toDouble()).$2.abs();
  }

  factory HoopFrame.fromJson(Map<String, dynamic> m) => HoopFrame(
        (m['left'] as List).cast<num>(),
        (m['right'] as List).cast<num>(),
        (m['front'] as List).cast<num>(),
      );

  (double, double) toHoop(double px, double py) {
    final tx = px - cx, ty = py - cy;
    return (
      (tx * _cos + ty * _sin) / radiusPx,
      (-tx * _sin + ty * _cos) / radiusPx,
    );
  }

  (double, double) toPixels(double x, double y) {
    final sx = x * radiusPx, sy = y * radiusPx;
    return (sx * _cos - sy * _sin + cx, sx * _sin + sy * _cos + cy);
  }

  double sizeInRimDiameters(double wPx, double hPx) =>
      ((wPx + hPx) / 2.0) / (2.0 * radiusPx);
}
