"""Back-compat shim for the old `rl_cages` import path.

The cage specialization moved to `ideal_poly_volume_toolkit.rl.cage.task`
during Stage C of the harness modularization.  All public symbols are
re-exported here so existing code — notably `fast_logdet.py` and any
scripts that `from ideal_poly_volume_toolkit.rl_cages import ...` —
keeps working.

New code should import from `ideal_poly_volume_toolkit.rl.cage`.
"""

from .rl.cage.task import *  # noqa: F401,F403
