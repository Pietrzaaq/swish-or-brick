"""Repository-relative paths, so tools and tests work from any working directory."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED = REPO_ROOT / "shared"
CONFIG_V1 = SHARED / "config" / "shot_logic.v1.json"
FIXTURE_CASES = SHARED / "fixtures" / "cases"
