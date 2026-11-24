"""ctypes bindings for the C ports of Chimp128 and Gorilla (fastcodecs.c).

The C ports produce the same byte streams as prismcodecs.py and the reference Java
implementation; they are used for all latency measurements.
"""
from __future__ import annotations

import ctypes
import os
from typing import Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "libfastcodecs.so")

_lib = ctypes.CDLL(LIB)
_lib.chimp128_encode.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_long,
                                 ctypes.POINTER(ctypes.c_uint8), ctypes.c_long]
_lib.chimp128_encode.restype = ctypes.c_long
_lib.chimp128_decode.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_long,
                                 ctypes.POINTER(ctypes.c_double), ctypes.c_long]
_lib.chimp128_decode.restype = ctypes.c_long
_lib.gorilla_encode.argtypes = _lib.chimp128_encode.argtypes
_lib.gorilla_encode.restype = ctypes.c_long
_lib.gorilla_decode.argtypes = _lib.chimp128_decode.argtypes
_lib.gorilla_decode.restype = ctypes.c_long
_lib.codec_max_bytes.argtypes = [ctypes.c_long]
_lib.codec_max_bytes.restype = ctypes.c_long


def _to_c_arr(values: Sequence[float]) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def chimp128_encode(values: Sequence[float]) -> bytes:
    arr = _to_c_arr(values)
    n = arr.size
    cap = int(_lib.codec_max_bytes(n))
    out = np.zeros(cap, dtype=np.uint8)
    bits = int(_lib.chimp128_encode(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                    n, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), cap))
    nbytes = (bits + 7) // 8
    return out[:nbytes].tobytes()


def chimp128_decode(data: bytes, max_values: int = 1 << 30) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    out = np.zeros(max_values, dtype=np.float64)
    n = int(_lib.chimp128_decode(raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                 raw.size, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 max_values))
    return out[:n]


def gorilla_encode(values: Sequence[float]) -> bytes:
    arr = _to_c_arr(values)
    n = arr.size
    cap = int(_lib.codec_max_bytes(n))
    out = np.zeros(cap, dtype=np.uint8)
    bits = int(_lib.gorilla_encode(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                   n, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), cap))
    nbytes = (bits + 7) // 8
    return out[:nbytes].tobytes()


def gorilla_decode(data: bytes, max_values: int = 1 << 30) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    out = np.zeros(max_values, dtype=np.float64)
    n = int(_lib.gorilla_decode(raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                 raw.size, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 max_values))
    return out[:n]
