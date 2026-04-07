import sys
import os

# Force the project root onto sys.path for all test files.
# os.path.abspath is required — dirname(__file__) alone returns "" when pytest
# is invoked from the project root directory, which is a no-op on sys.path.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
