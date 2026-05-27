import sys as _sys

from . import differential as _differential

_sys.modules.setdefault("differential", _differential)

from . import world as _world

_sys.modules.setdefault("world", _world)

Universe = _world.Universe

__all__ = ["Universe"]
