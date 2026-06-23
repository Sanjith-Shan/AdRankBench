"""Ensures the repository root is on sys.path so `import src` works under pytest
and when running scripts directly, without requiring an editable install.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
