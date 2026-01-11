import pytest


def test_decorate_imports_without_optional_deps():
    import galpos.decorate as dec

    assert hasattr(dec, "PYNBODY_AVAILABLE")
    assert hasattr(dec, "ANASTRISTNG_AVAILABLE")


def test_decorate_starbirth_missing_pynbody_gives_hint(monkeypatch: pytest.MonkeyPatch):
    import galpos.decorate as dec

    monkeypatch.setattr(dec, "PYNBODY_AVAILABLE", False)

    with pytest.raises(ImportError, match=r"pip install pynbody"):
        _ = dec.StarBirth

    with pytest.raises(ImportError, match=r"pip install pynbody"):
        _ = dec.make_star_birth


def test_decorate_tng_missing_anastristng_gives_hint(monkeypatch: pytest.MonkeyPatch):
    import galpos.decorate as dec

    monkeypatch.setattr(dec, "ANASTRISTNG_AVAILABLE", False)

    with pytest.raises(ImportError, match=r"AnastrisTNG"):
        _ = dec.make_tng_star_birth
