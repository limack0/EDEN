"""EDEN v2.0 — Emergent Developmental Encoding Network."""

from eden.config import SEED, AblationFlags, set_seed
from eden.inference import EDENPredictor

__version__ = "2.0.0"
__all__ = ["SEED", "AblationFlags", "set_seed", "EDENPredictor", "__version__"]
