"""Test configuration: make the ``src/`` layout importable.

The package ships in ``src/web_research/`` (PEP 517 ``src`` layout) but
Hatch installs it as the importable name ``web_research`` in editable mode.
This conftest adds ``src/`` to ``sys.path`` so tests can ``import web_research``
in either editable-install or ``PYTHONPATH=src`` execution modes, without
relying on the install step having happened first.
"""

import sys
from pathlib import Path

# Project root / src directory: <repo>/src
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
