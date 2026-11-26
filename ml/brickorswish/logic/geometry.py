"""Hoop frame: pixels <-> rim units, from the rim's left and right edge points."""

import math


class HoopFrame:
    """Rim units, used by all the logic: origin at the rim centre, 1 unit = rim radius, y along `down`, x
    perpendicular to it (to the right). So y < 0 is above the rim, |x| < 1 is inside it, and the same numbers work
    for any video size or distance from the hoop.

    `down` is the true vertical (image vertical for an upright phone, later the gravity sensor). Not the clicked
    rim's tilt, which is perspective and would make a straight fall drift sideways, nor the ball's apparent fall
    direction, which perspective also bends."""

    def __init__(self, left, right, front, down=(0.0, 1.0)):
        lx, ly = float(left[0]), float(left[1])
        rx, ry = float(right[0]), float(right[1])
        dx, dy = rx - lx, ry - ly
        length = math.hypot(dx, dy)
        if length <= 0.0:
            raise ValueError("left and right rim points must differ")
        gx, gy = float(down[0]), float(down[1])
        g = math.hypot(gx, gy)
        if g <= 0.0:
            raise ValueError("down must be a non-zero vector")
        self.cx = (lx + rx) / 2.0
        self.cy = (ly + ry) / 2.0
        self.radius_px = length / 2.0
        self.down = (gx / g, gy / g)
        # The rim as clicked, for drawing only; ellipse_b is the semi-minor axis in rim units.
        self.rim_angle = math.atan2(dy, dx)
        fx, fy = float(front[0]) - self.cx, float(front[1]) - self.cy
        self.ellipse_b = abs(fx * dy - fy * dx) / length / self.radius_px

    @classmethod
    def from_dict(cls, data: dict, down=(0.0, 1.0)) -> "HoopFrame":
        return cls(data["left"], data["right"], data["front"], down)

    def to_hoop(self, px: float, py: float) -> tuple[float, float]:
        tx = float(px) - self.cx
        ty = float(py) - self.cy
        gx, gy = self.down
        x = (tx * gy - ty * gx) / self.radius_px  # x axis = down rotated 90 degrees anticlockwise: (gy, -gx)
        y = (tx * gx + ty * gy) / self.radius_px
        return x, y

    def to_pixels(self, x: float, y: float) -> tuple[float, float]:
        gx, gy = self.down
        sx = x * self.radius_px
        sy = y * self.radius_px
        return sx * gy + sy * gx + self.cx, -sx * gx + sy * gy + self.cy

    def size_in_rim_diameters(self, w_px: float, h_px: float) -> float:
        return ((float(w_px) + float(h_px)) / 2.0) / (2.0 * self.radius_px)
