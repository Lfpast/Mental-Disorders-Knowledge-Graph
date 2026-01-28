"""
Optional third-party package imports used across the SynSpERT pipeline.

This module exposes optional packages (tensorboardX, jinja2) and falls back
to `None` when packages are unavailable so callers can gracefully degrade.
"""

# optional packages

try:
    import tensorboardX
except ImportError:
    tensorboardX = None


try:
    import jinja2
except ImportError:
    jinja2 = None
