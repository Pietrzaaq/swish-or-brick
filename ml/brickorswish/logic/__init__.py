"""Shot logic: pure functions, float64 only, no I/O inside process().

Everything in this package must port 1:1 to Dart, so: no numpy, no clocks, no randomness.
"""

from .config import ShotLogicConfig
from .geometry import HoopFrame
from .pipeline import process

__all__ = ["HoopFrame", "ShotLogicConfig", "process"]
