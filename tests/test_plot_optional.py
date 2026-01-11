import builtins
import sys

import numpy as np
import pytest

from galpos import GalaxyPoseTrajectory


def test_plot_requires_matplotlib_message(monkeypatch: pytest.MonkeyPatch):
    # Ensure any already-imported matplotlib modules are cleared so our import blocker works.
    for name in list(sys.modules.keys()):
        if name.startswith("matplotlib"):
            del sys.modules[name]

    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("matplotlib"):
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    t = np.array([0.0, 1.0, 2.0])
    pos = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]])
    vel = np.ones_like(pos)
    g = GalaxyPoseTrajectory(t, pos, vel)

    with pytest.raises(ImportError, match=r"pip install matplotlib"):
        g.plot()