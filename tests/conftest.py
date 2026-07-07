import os
import sys

# The bonsai/ and downstream_analyses/ packages are plain namespace packages
# (no __init__.py) importable only when the repo root is on sys.path. Make
# that true regardless of the current working directory pytest is invoked
# from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
