"""Optional integrations (lazy imports).

This subpackage provides utilities that depend on optional third-party libraries.
The core `galpos` package does not require these dependencies.

Optional features
-----------------
- `StarBirth` / `make_star_birth`: requires `pynbody`
- `make_tng_star_birth`: requires `AnastrisTNG` (+ `pynbody`)

Install hints
-------------
- `pip install pynbody`
- `pip install -e .[decorate]`
"""

from .util import PYNBODY_AVAILABLE, ANASTRISTNG_AVAILABLE


_PYNBODY_HINT = "galpos.decorate requires 'pynbody'. Install with: pip install pynbody"
_ANASTRISTNG_HINT = (
    "galpos.decorate.make_tng_star_birth requires AnastrisTNG. "
    "Install with: pip install git+https://github.com/wx-ys/AnastrisTNG"
)


def _import_starbirth():
    if not PYNBODY_AVAILABLE:
        raise ImportError(_PYNBODY_HINT)
    try:
        from .pynbody_decorate import StarBirth, make_star_birth
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or ""
        msg = str(e)
        if missing.startswith("pynbody") or ("pynbody" in msg):
            raise ImportError(_PYNBODY_HINT) from e
        raise

    return StarBirth, make_star_birth


def _import_anastristng():
    if not ANASTRISTNG_AVAILABLE:
        raise ImportError(_ANASTRISTNG_HINT)
    try:
        from .anastristng_decorate import make_star_birth as make_tng_star_birth
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or ""
        msg = str(e)
        if missing.startswith("AnastrisTNG") or ("AnastrisTNG" in msg):
            raise ImportError(_ANASTRISTNG_HINT) from e
        if missing.startswith("pynbody") or ("pynbody" in msg):
            raise ImportError(_PYNBODY_HINT) from e
        raise

    return make_tng_star_birth


def __getattr__(name: str) -> object:
    if name == "StarBirth":
        cls, _ = _import_starbirth()
        return cls
    if name == "make_star_birth":
        _, fn = _import_starbirth()
        return fn
    if name == "make_tng_star_birth":
        return _import_anastristng()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + ["StarBirth", "make_star_birth", "make_tng_star_birth"])


__all__ = ["StarBirth", "make_star_birth", "make_tng_star_birth"]