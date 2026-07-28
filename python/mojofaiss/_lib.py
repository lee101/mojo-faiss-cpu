"""ctypes bindings for the compiled Mojo search kernels."""

from __future__ import annotations

import ctypes
import os
import shlex
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.path.join(ROOT, "dist", "libmojo-faiss-cpu.so")

I = ctypes.c_int64
F = ctypes.c_float

_SIGNATURES = {
    "mf_flat_search": ([I] * 9, None),
    "mf_range_count": ([I] * 6 + [F, I], None),
    "mf_range_fill": ([I] * 8 + [F, I], None),
    "mf_assign": ([I] * 7, None),
    "mf_kmeans": ([I] * 9 + [F, I], F),
    "mf_residuals": ([I] * 6, None),
    "mf_ivf_flat_search": ([I] * 12, None),
    "mf_pq_encode": ([I] * 7, None),
    "mf_pq_decode": ([I] * 7, None),
    "mf_pq_tables": ([I] * 8, None),
    "mf_pq_scan": ([I] * 11, None),
    "mf_ivfpq_search": ([I] * 17, None),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOFAISS_MOJO")
    if override:
        return shlex.split(override)
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    manifest = os.path.join(ROOT, "pixi.toml")
    if os.path.exists(pixi) and os.path.exists(manifest):
        return [pixi, "run", "--manifest-path", manifest, "mojo"]
    raise BuildError("mojo not found; set MOJOFAISS_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(os.path.getmtime(path) for path in sources):
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        os.path.join(SRC, "capi.mojo"),
        "-o",
        LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIBRARY: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_LIBRARY, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _LIBRARY


def f32(x, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(x)
    if source.dtype != np.float32:
        raise TypeError("vectors must have dtype float32")
    if copy:
        return np.array(x, dtype=np.float32, order="C", copy=True)
    return np.ascontiguousarray(x)


def i64(x, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(x)
    if source.dtype != np.int64:
        raise TypeError("ids and labels must have dtype int64")
    if copy:
        return np.array(x, dtype=np.int64, order="C", copy=True)
    return np.ascontiguousarray(x)


def addr(x: np.ndarray) -> int:
    if not isinstance(x, np.ndarray) or not x.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous NumPy arrays")
    address = int(x.ctypes.data)
    if x.size and address == 0:
        raise RuntimeError("NumPy returned a null address for a non-empty buffer")
    return address


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
