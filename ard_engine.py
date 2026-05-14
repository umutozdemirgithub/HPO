"""Compatibility shim for the ARD engine.

The implementation now lives in ``ard.engine_core`` so the project can grow
without keeping every concern in a single top-level module. Existing imports
(``import ard_engine as eng``) continue to work unchanged, including access to
private helpers that the test suite exercises.
"""

from ard import engine_core as _engine_core

__all__ = [name for name in dir(_engine_core) if not name.startswith('__')]

globals().update({name: getattr(_engine_core, name) for name in __all__})
