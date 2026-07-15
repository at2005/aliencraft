import sys as _sys

from . import world as _world

_sys.modules.setdefault("world", _world)

AlienCraftWorld = _world.AlienCraftWorld

__all__ = ["AlienCraftWorld"]

try:
    from .env import AlienCraftEnv

    __all__.append("AlienCraftEnv")
except ModuleNotFoundError:
    # gymnasium is an optional dependency (install the "gym" extra)
    pass
