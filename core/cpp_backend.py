"""Lazy bridge to SIM_MARL, with auto path setup.

We try to locate the built extension in typical CMake output folders:
- cpp/build/Release (MSVC multi-config)
- cpp/build (single-config)

This allows running from different working directories without manually
setting PYTHONPATH.
"""

import importlib
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent

candidate_dirs = [
    root / "cpp" / "build" / "Release",
    root / "cpp" / "build" / "Debug",
    root / "cpp" / "build",
]

for d in candidate_dirs:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))


_cpp_mod = None


def _lazy_import():
    global _cpp_mod
    if _cpp_mod is None:
        try:
            _cpp_mod = importlib.import_module("DRIVESIMX_ENV")
        except ModuleNotFoundError:
            _cpp_mod = None
    return _cpp_mod


def has_cpp_backend():
    return _lazy_import() is not None


def _require():
    mod = _lazy_import()
    if mod is None:
        raise RuntimeError(
            "DriveSimX Environment backend not available – build it first (or ensure cpp/build/Release is on PYTHONPATH)."
        )
    return mod


def Car(*args, **kwargs):
    return _require().Car(*args, **kwargs)


def ScenarioEnv(*args, **kwargs):
    return _require().ScenarioEnv(*args, **kwargs)


def State(*args, **kwargs):
    return _require().State(*args, **kwargs)


def Lidar(*args, **kwargs):
    return _require().Lidar(*args, **kwargs)


# Expose C++ MCTS search
def mcts_search(*args, **kwargs):
    return _require().mcts_search(*args, **kwargs)


def mcts_search_lstm(*args, **kwargs):
    return _require().mcts_search_lstm(*args, **kwargs)


def mcts_search_lstm_torchscript(*args, **kwargs):
    return _require().mcts_search_lstm_torchscript(*args, **kwargs)


def mcts_search_tcn_torchscript_seq_to_shm(*args, **kwargs):
    return _require().mcts_search_tcn_torchscript_seq_to_shm(*args, **kwargs)
