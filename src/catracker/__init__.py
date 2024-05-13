import importlib.metadata as _metadata

__version__ = _metadata.version("catracker")

submodules = [
]

__all__ = [
    *submodules,
    "alignment",
    "export",
    "import",
    "models",
    "threshold",
    "velocity",
]


def __dir__():
    return __all__
