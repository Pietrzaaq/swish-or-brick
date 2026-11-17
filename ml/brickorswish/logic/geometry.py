"""Hoop frame: pixel <-> hoop-unit transform built from the 3-click calibration."""

import math


class HoopFrame:
    """Origin at rim centre, x from left to right rim edge, 1 unit = rim radius, y down."""

    def __init__(self, left, right, front):
        lx, ly = float(left[0]), float(left[1])
        rx, ry = float(right[0]), float(right[1])
        dx, dy = rx - lx, ry - ly
        length = math.hypot(dx, dy)
        if length <= 0.0:
            raise ValueError("left and right rim points must differ")
        self.cx = (lx + rx) / 2.0
        self.cy = (ly + ry) / 2.0
        self.radius_px = length / 2.0
        self._cos = dx / length
        self._sin = dy / length
        # Ellipse semi-minor axis in hoop units: distance of the front click from the rim axis.
        _, fy = self.to_hoop(front[0], front[1])
        self.ellipse_b = abs(fy)

    @classmethod
    def from_dict(cls, data: dict) -> "HoopFrame":
        return cls(data["left"], data["right"], data["front"])

    def to_hoop(self, px: float, py: float) -> tuple[float, float]:
        tx = float(px) - self.cx
        ty = float(py) - self.cy
        x = (tx * self._cos + ty * self._sin) / self.radius_px
        y = (-tx * self._sin + ty * self._cos) / self.radius_px
        return x, y

    def to_pixels(self, x: float, y: float) -> tuple[float, float]:
        sx = x * self.radius_px
        sy = y * self.radius_px
        px = sx * self._cos - sy * self._sin + self.cx
        py = sx * self._sin + sy * self._cos + self.cy
        return px, py

    def size_in_rim_diameters(self, w_px: float, h_px: float) -> float:
        return ((float(w_px) + float(h_px)) / 2.0) / (2.0 * self.radius_px)
