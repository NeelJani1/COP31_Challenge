"""Test runner script for COP31 Challenge."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    exit_code = pytest.main(["tests", "-v", "--tb=short"])
    sys.exit(exit_code)
