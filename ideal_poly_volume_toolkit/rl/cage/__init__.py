"""Cage (high-girth cubic graph) specialization of the RL harness.

Public API intentionally re-exports everything the old `rl_cages` module
exposed, so the top-level `rl_cages` back-compat shim can re-export the
same names.
"""

from .task import *  # noqa: F401,F403
