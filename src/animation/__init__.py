"""Animation module - animate still images with audio."""
from .replicate_backend import animate_with_replicate
from .local_backend import animate_local

__all__ = ["animate_with_replicate", "animate_local"]
