"""A general-purpose framework for planning on Riemannian manifolds."""

# Import standard modules under private names. The package exposes only geodex's own API.
import importlib as _importlib
import importlib.machinery as _machinery
import importlib.util as _importlib_util
import os as _os
import platform as _platform
import sys as _sys
import types as _types

__version__ = "1.0.0"


def _x86_64_v3_supported():
    """Return whether this x86-64 CPU and OS support AVX2 and FMA."""
    if _sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as cpuinfo:
                for line in cpuinfo:
                    if line.startswith("flags"):
                        return {"avx2", "fma"} <= set(line.split(":", 1)[1].split())
        except OSError:
            pass
        return False
    if _sys.platform == "darwin":
        import ctypes

        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")

        def has(name):
            value = ctypes.c_int(0)
            size = ctypes.c_size_t(ctypes.sizeof(value))
            status = libc.sysctlbyname(
                name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0
            )
            return status == 0 and value.value != 0

        return has("hw.optional.avx2_0") and has("hw.optional.fma")
    if _sys.platform == "win32":
        import ctypes

        return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
    return False


def _load_core():
    """Import the extension module, picking the instruction-set variant when there are several.

    Returns the module and the name of the variant it came from, or None for a build with a
    single module.
    """
    name = __name__ + "._geodex_core"
    if name in _sys.modules or _importlib_util.find_spec(name) is not None:
        return _importlib.import_module(name), None
    machine = _platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        raise ImportError(f"geodex has no extension module for this machine ({machine})")
    isa = "x86_64_v3" if _x86_64_v3_supported() else "x86_64"
    directory = _os.path.join(_os.path.dirname(__file__), "_isa", isa)
    for suffix in _machinery.EXTENSION_SUFFIXES:
        path = _os.path.join(directory, "_geodex_core" + suffix)
        if _os.path.exists(path):
            break
    else:
        raise ImportError(f"geodex extension module for {isa} is missing from {directory}")
    loader = _machinery.ExtensionFileLoader(name, path)
    spec = _importlib_util.spec_from_file_location(name, path, loader=loader)
    core = _importlib_util.module_from_spec(spec)
    _sys.modules[name] = core
    loader.exec_module(core)
    return core, isa


_geodex_core, _isa = _load_core()

from geodex._geodex_core import *  # noqa: E402, F401, F403

# Names that a build can lack raise an ImportError with a hint when used. Planning needs the
# OMPL fork. Collision scenes need VAMP, and VAMP needs AVX2 and FMA on x86-64. The robots,
# planners and vamp placeholders raise on attribute access.
_BUILD_HINT = (
    "geodex.{name} is not part of this geodex build, which was compiled without the "
    "planning stack (OMPL fork, VAMP and the robot catalog). The pygeodex wheels for "
    "Linux (manylinux) and macOS include it; see https://geodex.readthedocs.io for "
    "installation instructions."
)
_CPU_HINT = (
    "geodex.{name} needs VAMP collision checking, which requires an x86-64 CPU with AVX2 "
    "and FMA. This CPU lacks them, so geodex loaded its build without VAMP. Planning with "
    "a Python validity callable still works."
)
_COLLISION_HINT = _CPU_HINT if _isa == "x86_64" else _BUILD_HINT


def _missing_callable(name, hint):
    def _stub(*args, **kwargs):
        raise ImportError(hint.format(name=name))

    _stub.__name__ = name
    _stub.__qualname__ = name
    return _stub


class _MissingNamespace:
    def __init__(self, name, hint):
        self._name = name
        self._hint = hint

    def __getattr__(self, attr):
        raise ImportError(self._hint.format(name=self._name))


for _name, _hint, _namespace in (
    ("plan", _BUILD_HINT, False),
    ("PlanSettings", _BUILD_HINT, False),
    ("PlanResult", _BUILD_HINT, False),
    ("planners", _BUILD_HINT, True),
    ("robots", _BUILD_HINT, True),
    ("Scene", _COLLISION_HINT, False),
    ("load_scene", _COLLISION_HINT, False),
    ("vamp", _COLLISION_HINT, True),
):
    if not hasattr(_geodex_core, _name):
        globals()[_name] = (
            _MissingNamespace(_name, _hint) if _namespace else _missing_callable(_name, _hint)
        )

# Register the extension's submodules and placeholders in sys.modules under the package.
# `import geodex.robots` returns the submodule, not the stub-only directory next to this file.
for _name, _value in list(globals().items()):
    if isinstance(_value, _MissingNamespace) or (
        isinstance(_value, _types.ModuleType)
        and _value.__name__.startswith(_geodex_core.__name__ + ".")
    ):
        _sys.modules[f"{__name__}.{_name}"] = _value

del _name, _hint, _namespace, _value
