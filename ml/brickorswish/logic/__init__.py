"""Shot logic. Ported 1:1 to Dart, so: float64 only, no numpy, clocks, randomness or I/O inside process()."""

from .config import ShotLogicConfig
from .geometry import HoopFrame
from .pipeline import process

__all__ = ["HoopFrame", "ShotLogicConfig", "process"]
