from importlib.util import find_spec


__all__ = ["PYNBODY_AVAILABLE", "ANASTRISTNG_AVAILABLE"]



def module_available(name: str) -> bool:
    """Return True if a module is importable."""
    return find_spec(name) is not None

PYNBODY_AVAILABLE: bool = module_available("pynbody")
ANASTRISTNG_AVAILABLE: bool = module_available("AnastrisTNG")
