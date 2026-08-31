import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND = os.path.join(_ROOT, "backend")
for p in [_ROOT, _BACKEND]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.main import app
except ImportError:
    from main import app