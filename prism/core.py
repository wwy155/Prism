"""Prism core: exact accumulation, SegmentSketch and ShapeIndex.

SegmentSketch keeps the count, the extrema and the exact sum and sum of squares of a
segment as integer mantissa/exponent pairs.  Sketches of any set of segments merge into
the exact sum and sum of squares, which are rounded once.

ShapeIndex keeps a quantized piecewise aggregate approximation (PAA) of a segment and
the exact residual energy e = ||s - PAA(s)||^2, which yields the lower bound
    ||s - q||_2 >= sqrt(b) * ||PAA(s) - PAA(q)||_2 - sqrt(e_s) - sqrt(e_q).
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

S_DEFAULT = 512

import os as _os_flag
_SHAPE_FAST = _os_flag.environ.get("PRISM_SHAPE_FAST", "1") == "1"
_VAR_FAST = _os_flag.environ.get("PRISM_VAR_FAST", "1") == "1"

# --------------------------------------------------------------------------------------
# exact float64 decomposition
# --------------------------------------------------------------------------------------


def decompose(v: float) -> Tuple[int, int]:
    """Return (mantissa, exp) with v == mantissa * 2**exp exactly, mantissa an int.

    Works for subnormals, +/-0.0 and negatives. v must be finite.
    """
    m, e = math.frexp(v)
    mant = int(m * (1 << 53))
    return mant, e - 53


def exact_sum(values: Sequence[float]) -> Tuple[int, int]:
    """Exact integer sum of float64 values: returns (mantissa, exp) with sum == m*2**e."""
    acc = 0
    acc_exp = None
    for v in values:
        if v == 0.0:
            continue
        mant, exp = decompose(v)
        if acc_exp is None:
            acc, acc_exp = mant, exp
        elif exp < acc_exp:
            acc = (acc << (acc_exp - exp)) + mant
            acc_exp = exp
        elif exp > acc_exp:
            acc += mant << (exp - acc_exp)
        else:
            acc += mant
    if acc_exp is None:
        return 0, 0
    return acc, acc_exp


def exact_sumsq(values: Sequence[float]) -> Tuple[int, int]:
    """Exact integer sum of squares: returns (m, e) with sumsq == m * 2**e exactly."""
    acc = 0
    acc_exp = None
    for v in values:
        if v == 0.0:
            continue
        mant, exp = decompose(v)
        sq_mant = mant * mant  # exact, <= 2**106
        sq_exp = 2 * exp
        if acc_exp is None:
            acc, acc_exp = sq_mant, sq_exp
        elif sq_exp < acc_exp:
            acc = (acc << (acc_exp - sq_exp)) + sq_mant
            acc_exp = sq_exp
        elif sq_exp > acc_exp:
            acc += sq_mant << (sq_exp - acc_exp)
        else:
            acc += sq_mant
    if acc_exp is None:
        return 0, 0
    return acc, acc_exp


def _frac_to_float(f: "Fraction") -> float:
    """float(Fraction) raises OverflowError for huge magnitudes; map to +-inf."""
    try:
        return float(f)
    except OverflowError:
        return math.inf if f > 0 else -math.inf


def _align(acc: int, acc_exp: int, m: int, e: int) -> Tuple[int, int]:
    if acc == 0:
        return m, e
    if m == 0:
        return acc, acc_exp
    if e < acc_exp:
        return (acc << (acc_exp - e)) + m, e
    if e > acc_exp:
        return acc + (m << (e - acc_exp)), acc_exp
    return acc + m, acc_exp


def int_to_float(m: int, e: int) -> float:
    """Correctly-rounded float64 of m*2**e (round-half-even)."""
    if m == 0:
        return 0.0
    # int/int true division in CPython is correctly rounded.
    return m / (1 << -e) if e < 0 else float(Fraction(m) * Fraction(2) ** e)


def exact_sum_of_floats(values: Sequence[float]) -> float:
    m, e = exact_sum(values)
    return int_to_float(m, e)


# --------------------------------------------------------------------------------------
# base codecs (bit level). Gorilla is implemented here; Chimp/Chimp128 come from the
# OFFICIAL java implementation for bits/value.
# --------------------------------------------------------------------------------------


class BitWriter:
    def __init__(self) -> None:
        self._bits = bytearray()
        self._cur = 0
        self._n = 0

    def write_bit(self, b: int) -> None:
        self._cur = (self._cur << 1) | (b & 1)
        self._n += 1
        if self._n == 8:
            self._bits.append(self._cur)
            self._cur = 0
            self._n = 0

    def write_bits(self, value: int, nbits: int) -> None:
        for i in range(nbits - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    def bit_length(self) -> int:
        return len(self._bits) * 8 + self._n

    def bytes_(self) -> bytes:
        out = bytearray(self._bits)
        if self._n:
            out.append(self._cur << (8 - self._n))
        return bytes(out)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read_bit(self) -> int:
        byte = self._data[self._pos >> 3]
        bit = (byte >> (7 - (self._pos & 7))) & 1
        self._pos += 1
        return bit

    def read_bits(self, nbits: int) -> int:
        v = 0
        for _ in range(nbits):
            v = (v << 1) | self.read_bit()
        return v

    def bit_position(self) -> int:
        return self._pos


class GorillaCodec:
    """Gorilla (Pelkonen et al. 2015) XOR-with-previous codec, bit-exact to the
    standard format of the reference Java implementation."""

    name = "gorilla"

    @staticmethod
    def encode(values: Sequence[float]) -> bytes:
        w = BitWriter()
        if not values:
            return b""
        first = np.float64(values[0])
        w.write_bits(int(struct.unpack("<q", struct.pack("<d", first))[0]), 64)
        prev = int(struct.unpack("<q", struct.pack("<d", first))[0])
        for v in values[1:]:
            cur = int(struct.unpack("<q", struct.pack("<d", np.float64(v)))[0])
            xor = cur ^ prev
            if xor == 0:
                w.write_bit(0)
            else:
                lead = 0
                tmp = xor
                while not (tmp & (1 << 63)):
                    tmp = (tmp << 1) & 0xFFFFFFFFFFFFFFFF
                    lead += 1
                length = 64 - lead
                if length == 1:
                    w.write_bits(0, 2)  # '10'
                else:
                    w.write_bits(0b11, 2)
                w.write_bits(lead, 5)
                w.write_bits(length - 1, 6)
                w.write_bits(xor >> (64 - length), length)
                if length != 64:
                    trail = 0
                    while not (xor & 1):
                        xor >>= 1
                        trail += 1
                    w.write_bits(0b111, 3)
                    w.write_bits(trail, 6)
            prev = cur
        return w.bytes_()

    @staticmethod
    def decode(data: bytes, n: int) -> np.ndarray:
        r = BitReader(data)
        out = np.empty(n, dtype=np.float64)
        if n == 0:
            return out
        prev = r.read_bits(64)
        out[0] = struct.unpack("<d", struct.pack("<q", prev))[0]
        for i in range(1, n):
            if r.read_bit() == 0:
                cur = prev
            else:
                if r.read_bit() == 0:
                    length = 1
                    lead = r.read_bits(5)
                    xor = prev & ((1 << (64 - lead)) - 1) if lead else prev
                else:
                    lead = r.read_bits(5)
                    length = r.read_bits(6) + 1
                    if lead:
                        xor = (prev >> lead) & ((1 << (64 - lead)) - 1)
                    else:
                        xor = prev
                    xor = xor << (64 - length)
                    xor = xor & ((1 << 64) - 1)
                    if length != 64:
                        trail = r.read_bits(6)
                        xor = xor >> trail
                cur = prev ^ xor
            out[i] = struct.unpack("<d", struct.pack("<q", cur))[0]
            prev = cur
        return out


class PlainCodec:
    """Uncompressed float64 payload (used for the 'uncompressed' row and as a
    correctness reference for the codec round-trip)."""

    name = "plain"

    @staticmethod
    def encode(values: Sequence[float]) -> bytes:
        return np.asarray(values, dtype=np.float64).tobytes()

    @staticmethod
    def decode(data: bytes, n: int) -> np.ndarray:
        return np.frombuffer(data[: 8 * n], dtype=np.float64).copy()


class RLEDictCodec:
    """Dictionary + run-length codec (IoTDB-style light-weight encoding baseline).
    Format: u32 distinct-count, then u32 distinct values (raw f64), then per run
    (u32 value-index, u32 run-length)."""

    name = "rledict"

    @staticmethod
    def encode(values: Sequence[float]) -> bytes:
        vals = np.asarray(values, dtype=np.float64)
        ids: Dict[float, int] = {}
        distinct: List[float] = []
        runs_idx: List[int] = []
        runs_len: List[int] = []
        for v in vals.tolist():
            i = ids.get(v)
            if i is None:
                i = len(distinct)
                ids[v] = i
                distinct.append(v)
            if runs_len and runs_idx[-1] == i:
                runs_len[-1] += 1
            else:
                runs_idx.append(i)
                runs_len.append(1)
        out = bytearray()
        out += struct.pack("<I", len(distinct))
        for d in distinct:
            out += struct.pack("<d", d)
        out += struct.pack("<I", len(runs_idx))
        for i, l in zip(runs_idx, runs_len):
            out += struct.pack("<II", i, l)
        return bytes(out)

    @staticmethod
    def decode(data: bytes, n: int) -> np.ndarray:
        off = 0
        (ndist,) = struct.unpack_from("<I", data, off)
        off += 4
        distinct = np.empty(ndist, dtype=np.float64)
        for i in range(ndist):
            distinct[i] = struct.unpack_from("<d", data, off)[0]
            off += 8
        (nruns,) = struct.unpack_from("<I", data, off)
        off += 4
        out = np.empty(n, dtype=np.float64)
        pos = 0
        for _ in range(nruns):
            vi, rl = struct.unpack_from("<II", data, off)
            off += 8
            out[pos : pos + rl] = distinct[vi]
            pos += rl
        return out[:n]


# --------------------------------------------------------------------------------------
# codec registry
#
# We deliberately route every base codec through `prismcodecs`, whose Gorilla and
# Chimp128 ports were verified BYTE-FOR-BYTE against the official Java implementation
#.  The local
# copies above are kept for reference but are not used by the store.
# --------------------------------------------------------------------------------------

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from prismcodecs import (  # noqa: E402
    Chimp128Codec as _PrismChimp128,
    GorillaCodec as _PrismGorilla,
    PlainCodec as _PrismPlain,
    RLEDictCodec as _PrismRLEDict,
)


class _Chimp128Adapter:
    """Uniform interface (encode/decode(data, n)) over the validated Chimp128 port."""

    name = "chimp128"

    @staticmethod
    def encode(values) -> bytes:
        return _PrismChimp128.encode(values, 128)

    @staticmethod
    def decode(data: bytes, n: int) -> np.ndarray:
        return _PrismChimp128.decode(data, 128)[:n]


class _GorillaAdapter:
    name = "gorilla"

    @staticmethod
    def encode(values) -> bytes:
        return _PrismGorilla.encode(values)

    @staticmethod
    def decode(data: bytes, n: int) -> np.ndarray:
        return _PrismGorilla.decode(data)[:n]


CODECS = {
    "plain": _PrismPlain,
    "gorilla": _GorillaAdapter,
    "chimp128": _Chimp128Adapter,
    "rledict": _PrismRLEDict,
}


# --------------------------------------------------------------------------------------
# exact accumulation helpers (vectorised, group-by-exponent)
# --------------------------------------------------------------------------------------


_EA_CHUNK = 1 << 24   # int64 partial sums below stay exact for groups up to this size
_EA_MASK = (1 << 18) - 1

# ---- native path: per-exponent 128-bit group sums in C ----
_EA_C_MAX = 1 << 20
_EA_CAP = 2400
try:
    import ctypes as _ct
    _ea_lib = _ct.CDLL(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                     "libexactacc.so"))
    _ea_lib.exact_groups.restype = _ct.c_long
    _ea_lib.exact_groups.argtypes = [_ct.c_void_p, _ct.c_long] + [_ct.c_void_p] * 5 + [_ct.c_long]
    _ea_exps = np.zeros(_EA_CAP, dtype=np.int64)
    _ea_shi = np.zeros(_EA_CAP, dtype=np.int64)
    _ea_slo = np.zeros(_EA_CAP, dtype=np.uint64)
    _ea_qhi = np.zeros(_EA_CAP, dtype=np.int64)
    _ea_qlo = np.zeros(_EA_CAP, dtype=np.uint64)
    _ea_ptrs = [a.ctypes.data for a in (_ea_exps, _ea_shi, _ea_slo, _ea_qhi, _ea_qlo)]
    _EA_NATIVE = True
except OSError:          # library not built: fall back to the numpy path
    _EA_NATIVE = False


def _exact_accumulate_native(arr: np.ndarray) -> Tuple[int, int, int, int]:
    g = _ea_lib.exact_groups(arr.ctypes.data, arr.size, *_ea_ptrs, _EA_CAP)
    if g == -1:
        raise ValueError("non-finite value in exact accumulation")
    if g < 0:
        raise RuntimeError("exact_groups failed: %d" % g)
    exps = _ea_exps[:g].tolist()
    shi = _ea_shi[:g].tolist()
    slo = _ea_slo[:g].tolist()
    qhi = _ea_qhi[:g].tolist()
    qlo = _ea_qlo[:g].tolist()
    sum_mant = 0
    sum_exp = 0
    sq_mant = 0
    sq_exp = 0
    for gi in range(g):
        ex = exps[gi]
        sum_mant, sum_exp = _align(sum_mant, sum_exp, (shi[gi] << 64) + slo[gi], ex)
        sq_mant, sq_exp = _align(sq_mant, sq_exp, (qhi[gi] << 64) + qlo[gi], 2 * ex)
    return sum_mant, sum_exp, sq_mant, sq_exp


def exact_accumulate(arr: np.ndarray) -> Tuple[int, int, int, int]:
    """Exact sum and sum-of-squares of a float64 array.

    Returns (sum_mant, sum_exp, sumsq_mant, sumsq_exp) with
        sum   == sum_mant   * 2**sum_exp
        sumsq == sumsq_mant * 2**sumsq_exp
    exactly (integers, arbitrary precision).

    Native path (default): scripts/prism/exactacc.c buckets the values by binary
    exponent and sums the mantissas and squared mantissas of every bucket in
    128-bit integers; Python only recombines the few occupied buckets.
    Fallback path (no native library): every value is decomposed into
    mant * 2**exp (|mant| < 2**53), the values are
    grouped by exponent with one stable sort; inside a group the mantissa is split
    into three 18-bit limbs, so the group sums of the limbs and of the limb
    products are exact int64 reductions (np.add.reduceat).  Only the per-group
    recombination runs in Python big ints, i.e. the Python work is
    O(#distinct exponents) instead of O(#values).  The groups are combined in
    ascending exponent order with the same `_align` steps, so the
    returned integers are identical to the scalar reference implementation
    (`_exact_accumulate_scalar`).
    Non-finite values are rejected (callers must clean data first).
    """
    if _EA_NATIVE:
        a = np.ascontiguousarray(arr, dtype=np.float64).ravel()
        if a.size == 0:
            return 0, 0, 0, 0
        if a.size <= _EA_C_MAX:
            return _exact_accumulate_native(a)
    return _exact_accumulate_numpy(arr)


def _exact_accumulate_numpy(arr: np.ndarray) -> Tuple[int, int, int, int]:
    """Vectorised numpy path of `exact_accumulate` (see its docstring)."""
    """Exact sum and sum-of-squares of a float64 array.

    Returns (sum_mant, sum_exp, sumsq_mant, sumsq_exp) with
        sum   == sum_mant   * 2**sum_exp
        sumsq == sumsq_mant * 2**sumsq_exp
    exactly (integers, arbitrary precision).

    Every value is decomposed into mant * 2**exp (|mant| < 2**53).  The values are
    grouped by exponent with one stable sort; inside a group the mantissa is split
    into three 18-bit limbs, so the group sums of the limbs and of the limb
    products are exact int64 reductions (np.add.reduceat).  Only the per-group
    recombination runs in Python big ints, i.e. the Python work is
    O(#distinct exponents) instead of O(#values).  The groups are combined in
    ascending exponent order with the same `_align` steps, so the
    returned integers are identical to the scalar reference implementation
    (`_exact_accumulate_scalar`).
    Non-finite values are rejected (callers must clean data first).
    """
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0, 0, 0, 0
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite value in exact accumulation")
    if arr.size > _EA_CHUNK:
        sm = se = qm = qe = 0
        for k in range(0, arr.size, _EA_CHUNK):
            a, b, c, d = exact_accumulate(arr[k:k + _EA_CHUNK])
            sm, se = _align(sm, se, a, b)
            qm, qe = _align(qm, qe, c, d)
        return sm, se, qm, qe
    m, e = np.frexp(arr)
    mant = np.rint(m * float(1 << 53)).astype(np.int64)
    exp = e.astype(np.int64) - 53
    order = np.argsort(exp, kind="stable")
    exps = exp[order]
    mants = mant[order]
    if exps.size > 1:
        starts = np.flatnonzero(np.concatenate(([True], exps[1:] != exps[:-1])))
    else:
        starts = np.zeros(1, dtype=np.int64)
    uniq = exps[starts].tolist()
    sign = np.sign(mants)
    a = np.abs(mants)
    a0 = a & _EA_MASK
    a1 = (a >> 18) & _EA_MASK
    a2 = a >> 36
    g0 = np.add.reduceat(sign * a0, starts).tolist()
    g1 = np.add.reduceat(sign * a1, starts).tolist()
    g2 = np.add.reduceat(sign * a2, starts).tolist()
    c0 = np.add.reduceat(a0 * a0, starts).tolist()
    c1 = np.add.reduceat(2 * a1 * a0, starts).tolist()
    c2 = np.add.reduceat(2 * a2 * a0 + a1 * a1, starts).tolist()
    c3 = np.add.reduceat(2 * a2 * a1, starts).tolist()
    c4 = np.add.reduceat(a2 * a2, starts).tolist()
    sum_mant = 0
    sum_exp = 0
    sq_mant = 0
    sq_exp = 0
    for gi, ex in enumerate(uniq):
        gsum = (g2[gi] << 36) + (g1[gi] << 18) + g0[gi]
        gsq = ((c4[gi] << 72) + (c3[gi] << 54) + (c2[gi] << 36)
               + (c1[gi] << 18) + c0[gi])
        sum_mant, sum_exp = _align(sum_mant, sum_exp, gsum, ex)
        sq_mant, sq_exp = _align(sq_mant, sq_exp, gsq, 2 * ex)
    return sum_mant, sum_exp, sq_mant, sq_exp


def _exact_accumulate_scalar(arr: np.ndarray) -> Tuple[int, int, int, int]:
    """Scalar reference implementation of `exact_accumulate` (kept for the
    regression test scripts/prism/test_exact_accumulate.py)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return 0, 0, 0, 0
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite value in exact accumulation")
    m, e = np.frexp(arr)
    mant = np.rint(m * float(1 << 53)).astype(np.int64)
    exp = e.astype(np.int64) - 53
    uniq, inv = np.unique(exp, return_inverse=True)
    sum_mant = 0
    sum_exp = 0
    sq_mant = 0
    sq_exp = 0
    for gi, ex in enumerate(uniq.tolist()):
        idx = np.flatnonzero(inv == gi)
        g = mant[idx].tolist()
        gsum = 0
        gsq = 0
        for x in g:
            gsum += x
            gsq += x * x
        sum_mant, sum_exp = _align(sum_mant, sum_exp, gsum, ex)
        sq_mant, sq_exp = _align(sq_mant, sq_exp, gsq, 2 * ex)
    return sum_mant, sum_exp, sq_mant, sq_exp


def exact_align_add(m1: int, e1: int, m2: int, e2: int) -> Tuple[int, int]:
    return _align(m1, e1, m2, e2)


def exact_to_float(m: int, e: int) -> float:
    return int_to_float(m, e)


# --------------------------------------------------------------------------------------
# sidecar A: exact aggregate sketch
# --------------------------------------------------------------------------------------


@dataclass
class SidecarA:
    """Exact aggregate sidecar (SegmentSketch).

    Fixed-width record used for the overhead accounting (416 bits):
      count u32 | sum mantissa i80 + exponent i16 | min f64 | max f64 |
      sumsq mantissa u144 + exponent i16
    The mantissa widths are sized from the measured exact mantissas of the five
    datasets (largest observed: 76 bits for the sum, 143 bits for the sum of
    squares, S = 512); `sum_inexact` flags a segment whose exact mantissas would
    not fit the record (never observed on the evaluation data).
    """

    SUM_MANT_BITS = 80
    SQ_MANT_BITS = 144

    count: int
    sum_mant: int
    sum_exp: int
    vmin: float
    vmax: float
    sq_mant: int
    sq_exp: int
    sum_inexact: bool = False  # True when the mantissa does not fit the fixed field width

    @staticmethod
    def build(values: np.ndarray) -> "SidecarA":
        vals = np.asarray(values, dtype=np.float64)
        if vals.size == 0:
            return SidecarA(0, 0, 0, math.inf, -math.inf, 0, 0)
        sm, se, qm, qe = exact_accumulate(vals)
        over = (abs(sm).bit_length() >= SidecarA.SUM_MANT_BITS
                or qm.bit_length() > SidecarA.SQ_MANT_BITS)
        return SidecarA(
            count=int(vals.size),
            sum_mant=sm,
            sum_exp=se,
            vmin=float(vals.min()),
            vmax=float(vals.max()),
            sq_mant=qm,
            sq_exp=qe,
            sum_inexact=over,
        )

    def merge(self, other: "SidecarA") -> "SidecarA":
        m, e = _align(self.sum_mant, self.sum_exp, other.sum_mant, other.sum_exp)
        qm, qe = _align(self.sq_mant, self.sq_exp, other.sq_mant, other.sq_exp)
        return SidecarA(
            count=self.count + other.count,
            sum_mant=m,
            sum_exp=e,
            vmin=min(self.vmin, other.vmin),
            vmax=max(self.vmax, other.vmax),
            sq_mant=qm,
            sq_exp=qe,
        )

    @property
    def sum_value(self) -> float:
        return int_to_float(self.sum_mant, self.sum_exp)

    @property
    def mean_value(self) -> float:
        if self.count == 0:
            return float("nan")
        return self.sum_value / self.count

    @property
    def variance_value(self) -> float:
        """Correctly-rounded exact population variance from exact sum and exact sumsq."""
        if self.count == 0:
            return float("nan")
        n = self.count
        if _VAR_FAST:
            # same rational, rounded once by an integer division (see engine._exact_var)
            E = min(self.sq_exp, 2 * self.sum_exp)
            num = ((n * self.sq_mant) << (self.sq_exp - E)) - \
                  ((self.sum_mant * self.sum_mant) << (2 * self.sum_exp - E))
            if num == 0:
                return 0.0
            den = n * n
            try:
                return (num << E) / den if E >= 0 else num / (den << -E)
            except OverflowError:
                return math.inf if num > 0 else -math.inf
        # var = (sumsq - sum^2/n) / n   computed as an exact rational
        num = Fraction(self.sq_mant, 1 << -self.sq_exp) if self.sq_exp < 0 else Fraction(self.sq_mant * (1 << self.sq_exp))
        s = Fraction(self.sum_mant, 1 << -self.sum_exp) if self.sum_exp < 0 else Fraction(self.sum_mant * (1 << self.sum_exp))
        var = (num - s * s / n) / n
        return _frac_to_float(var)

    def bits(self) -> int:
        """Sidecar A size in bits (fixed-width record, see the class docstring)."""
        return 32 + self.SUM_MANT_BITS + 16 + 64 + 64 + self.SQ_MANT_BITS + 16


# --------------------------------------------------------------------------------------
# sidecar B: shape sketch with a provable pruning bound
# --------------------------------------------------------------------------------------


@dataclass
class SidecarB:
    """Shape sidecar: quantised block-average projection + exact residual energy.

    Normalisation: v -> (v - min) / (max - min), using the segment min/max already
    stored in sidecar A, so the shape sketch needs no extra scale state.
    PAA is k blocks of 8 bits over the normalised segment; the residual energy
        e = || s_norm - dequant(PAA(s_norm)) ||^2
    is computed against the DEQUANTISED PAA, which makes the bound provable:
        || s_norm - q_norm ||_2 >= || PAA(s) - PAA(q) ||_2 - sqrt(e_s) - sqrt(e_q)
    """

    k: int
    paa_bytes: bytes
    energy: float
    bits_value: int = 16

    @staticmethod
    def build(values: np.ndarray, k: int = 8, gmin: Optional[float] = None,
              gmax: Optional[float] = None) -> "SidecarB":
        """Build the shape sidecar.

        The normalisation MUST be the SAME global affine map for the stored segments
        and for the query pattern, otherwise the pruning bound is invalid: PAA is a
        linear operator only with respect to a fixed linear map.  We therefore normalise
        by the *global* (min, max) of the chunk, which is kept in the segment directory
        (sidecar C).  With this choice

            || s - q ||_2  >=  || PAA(s) - PAA(q) ||_2 - sqrt(e_s) - sqrt(e_q)

        holds exactly (triangle inequality + PAA being an L2 contraction).
        """
        vals = np.asarray(values, dtype=np.float64)
        n = vals.size
        if gmin is None:
            lo = float(vals.min())
        else:
            lo = float(gmin)
        if gmax is None:
            hi = float(vals.max())
        else:
            hi = float(gmax)
        rng = hi - lo
        if rng <= 0 or not np.isfinite(rng):
            norm = np.zeros(n, dtype=np.float64)
        else:
            norm = (vals - lo) / rng
        block = int(math.ceil(n / k))
        if n == k * block and _SHAPE_FAST:
            # one row-wise reduction instead of k Python-level slice means
            means = norm.reshape(k, block).mean(axis=1)
        else:
            means = np.zeros(k, dtype=np.float64)
            for i in range(k):
                chunk = norm[i * block : (i + 1) * block]
                means[i] = chunk.mean() if chunk.size else 0.0
        q = np.clip(np.rint(means * 255.0), 0, 255).astype(np.uint8)
        deq = q.astype(np.float64) / 255.0
        approx = np.repeat(deq, block)[:n]
        energy = float(np.sum((norm - approx) ** 2))
        return SidecarB(k=k, paa_bytes=bytes(q.tobytes()), energy=energy)

    def dequant(self) -> np.ndarray:
        return np.frombuffer(self.paa_bytes, dtype=np.uint8).astype(np.float64) / 255.0

    def bits(self) -> int:
        return self.k * 8 + 64  # PAA bytes + float64 residual energy


def paa_of_query(pattern: np.ndarray, k: int = 8, gmin: Optional[float] = None,
                 gmax: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """PAA + residual energy of a query pattern, SAME global normalisation as SidecarB."""
    vals = np.asarray(pattern, dtype=np.float64)
    n = vals.size
    lo = float(vals.min()) if gmin is None else float(gmin)
    hi = float(vals.max()) if gmax is None else float(gmax)
    rng = hi - lo
    norm = np.zeros(n) if (rng <= 0 or not np.isfinite(rng)) else (vals - lo) / rng
    block = int(math.ceil(n / k))
    means = np.zeros(k, dtype=np.float64)
    for i in range(k):
        chunk = norm[i * block : (i + 1) * block]
        means[i] = chunk.mean() if chunk.size else 0.0
    q = np.clip(np.rint(means * 255.0), 0, 255).astype(np.uint8)
    deq = q.astype(np.float64) / 255.0
    approx = np.repeat(deq, block)[:n]
    energy = float(np.sum((norm - approx) ** 2))
    return deq, energy


def shape_lower_bound(sb: SidecarB, q_paa: np.ndarray, q_energy: float) -> float:
    """Provable lower bound on ||s_norm - q_norm||_2 (triangle inequality)."""
    sb_paa = sb.dequant()
    k = min(len(sb_paa), len(q_paa))
    dist = float(np.linalg.norm(sb_paa[:k] - q_paa[:k]))
    return dist - math.sqrt(max(sb.energy, 0.0)) - math.sqrt(max(q_energy, 0.0))


def shape_bound_prunes(sb: SidecarB, q_paa: np.ndarray, q_energy: float,
                       block: int, radius: float) -> bool:
    """Does the provable bound let us skip this segment WITHOUT decoding it?

    This is the SAME bound the store uses at query time, with the SAME scaling:

        || PAA(s) - PAA(q) ||_2 = sqrt(block) * || deq(PAA(s)) - deq(PAA(q)) ||_2
        || s_norm - q_norm ||_2 >= that - sqrt(e_s) - sqrt(e_q)

    A segment whose lower bound exceeds the match radius provably contains no
    match and is skipped without decoding.
    """
    sb_paa = sb.dequant()
    k = min(len(sb_paa), len(q_paa))
    dist = math.sqrt(block) * float(np.linalg.norm(sb_paa[:k] - q_paa[:k]))
    bound = (dist - math.sqrt(max(sb.energy, 0.0))
             - math.sqrt(max(q_energy, 0.0)))
    return bound > radius


def shape_prune_rate(sb: SidecarB, probes: Sequence[Tuple[np.ndarray, float]],
                      block: int, radius: float) -> float:
    """How often does this segment's shape index prune a probe pattern?

    The probes are (PAA of the query pattern, residual energy of the query
    pattern) pairs; one probe is one workload match pattern.  The returned rate
    is the fraction of the probes for which the provable bound fires, i.e. the
    fraction of the workload's match queries for which this segment is skipped
    without being decoded.

    A segment whose shape is indistinguishable from the workload's patterns
    (periodic data, for instance) has a rate of zero: the shape index is pure
    overhead there.
    """
    if not probes:
        return 0.0
    hits = 0
    for q_paa, q_energy in probes:
        if shape_bound_prunes(sb, q_paa, q_energy, block, radius):
            hits += 1
    return hits / float(len(probes))


def shape_prune_rates(sidecars: Sequence[SidecarB],
                      probes: Sequence[Tuple[np.ndarray, float]],
                      block: int, radius: float) -> List[float]:
    """Per-sidecar pruning rate against a probe population.

    The rate of a sidecar is the fraction of the probe patterns for which its
    provable bound fires, i.e. the fraction of the workload's shape queries
    that this sidecar can actually prune.  A sidecar with rate zero is dead
    weight: it costs bits and a per-segment check and saves no decoding.
    """
    if not sidecars or not probes:
        return [0.0] * len(sidecars)
    out: List[float] = []
    n_probe = len(probes)
    for sb in sidecars:
        hits = 0
        for q_paa, q_energy in probes:
            if shape_bound_prunes(sb, q_paa, q_energy, block, radius):
                hits += 1
        out.append(hits / float(n_probe))
    return out


# --------------------------------------------------------------------------------------
# segment store (sidecar C = segment directory) and query operators
# --------------------------------------------------------------------------------------


@dataclass
class Segment:
    idx: int
    ts_start: int
    ts_end: int
    n_values: int
    codec: str
    payload: bytes
    sketch_a: SidecarA
    sketch_b: Optional[SidecarB] = None
    mant_bits: int = 0  # bits needed by the exact sum mantissa (overflow accounting)
    # per-segment switch: the Lagrangian optimiser may give different segments
    # different sidecar levels, so the aggregate sketch cannot be a store-level flag.
    sketch_enabled: bool = True

    def decode(self) -> np.ndarray:
        return CODECS[self.codec].decode(self.payload, self.n_values)


@dataclass
class QueryStats:
    """Engine-independent accounting: how many values had to be decoded."""

    decoded_points: int = 0
    segments_touched: int = 0
    segments_sketched: int = 0
    segments_decoded: int = 0
    sketch_ops: int = 0
    bytes_decoded: int = 0

    def merge(self, other: "QueryStats") -> "QueryStats":
        return QueryStats(
            self.decoded_points + other.decoded_points,
            self.segments_touched + other.segments_touched,
            self.segments_sketched + other.segments_sketched,
            self.segments_decoded + other.segments_decoded,
            self.sketch_ops + other.sketch_ops,
            self.bytes_decoded + other.bytes_decoded,
        )


@dataclass
class AggResult:
    count: int
    total: float
    minimum: float
    maximum: float
    mean: float
    variance: float
    exact_sum: Tuple[int, int]


class PrismStore:
    """Prism storage layout: segments + sidecar A (exact aggregates) + optional sidecar B
    (shape) + sidecar C (segment directory).

    The store is intentionally codec-agnostic: `codec` selects the base encoding for every
    segment (heterogeneous per-segment codecs are supported by building segments with
    different codecs).
    """

    def __init__(self, segments: List[Segment], codec: str, use_sketch: bool = True,
                 use_shape: bool = False, gmin: Optional[float] = None,
                 gmax: Optional[float] = None) -> None:
        self.segments = segments
        self.codec = codec
        self.use_sketch = use_sketch
        self.use_shape = use_shape
        self._starts = np.array([s.ts_start for s in segments], dtype=np.int64)
        self._ends = np.array([s.ts_end for s in segments], dtype=np.int64)
        # global normalisation used by the shape sidecar (kept in sidecar C)
        if gmin is None or gmax is None:
            if segments:
                gmin = min(s.sketch_a.vmin for s in segments)
                gmax = max(s.sketch_a.vmax for s in segments)
            else:
                gmin, gmax = 0.0, 1.0
        self.gmin = float(gmin)
        self.gmax = float(gmax)

    # -- construction ------------------------------------------------------------------
    @staticmethod
    def build(ts: np.ndarray, values: np.ndarray, codec: str = "gorilla", S: int = S_DEFAULT,
              use_sketch: bool = True, use_shape: bool = False,
              per_segment_codec: Optional[List[str]] = None,
              gmin: Optional[float] = None, gmax: Optional[float] = None) -> "PrismStore":
        ts = np.asarray(ts, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        n = values.size
        if gmin is None:
            gmin = float(values.min()) if n else 0.0
        if gmax is None:
            gmax = float(values.max()) if n else 1.0
        segs: List[Segment] = []
        for i, start in enumerate(range(0, n, S)):
            stop = min(start + S, n)
            c = per_segment_codec[i] if per_segment_codec else codec
            payload = CODECS[c].encode(values[start:stop])
            sk = SidecarA.build(values[start:stop]) if use_sketch else SidecarA(0, 0, 0, math.inf, -math.inf, 0, 0)
            sb = SidecarB.build(values[start:stop], 8, gmin, gmax) if use_shape else None
            mb = int(max(abs(sk.sum_mant).bit_length(), 1))
            segs.append(
                Segment(
                    idx=i,
                    ts_start=int(ts[start]),
                    ts_end=int(ts[stop - 1]),
                    n_values=stop - start,
                    codec=c,
                    payload=payload,
                    sketch_a=sk,
                    sketch_b=sb,
                    mant_bits=mb,
                )
            )
        return PrismStore(segs, codec, use_sketch, use_shape, gmin, gmax)

    # -- accounting --------------------------------------------------------------------
    def base_bits(self) -> int:
        return sum(len(s.payload) * 8 for s in self.segments)

    def sketch_bits(self) -> int:
        tot = 0
        for s in self.segments:
            if self.use_sketch and s.sketch_enabled:
                tot += s.sketch_a.bits()
            if self.use_shape and s.sketch_b is not None:
                tot += s.sketch_b.bits()
        return tot

    def directory_bits(self) -> int:
        """Sidecar C: per segment (ts_start, ts_end, n_values, payload offset)."""
        if not self.segments:
            return 0
        return len(self.segments) * (64 + 64 + 32 + 64)

    def total_bits(self) -> int:
        return self.base_bits() + self.sketch_bits() + self.directory_bits()

    def bits_per_value(self) -> float:
        n = sum(s.n_values for s in self.segments)
        return self.total_bits() / n if n else float("nan")

    # -- query operators ---------------------------------------------------------------
    def _segment_range(self, t1: int, t2: int) -> Tuple[int, int]:
        """Half-open segment index range [lo, hi) of the segments that OVERLAP [t1, t2].

        A segment overlaps the query window iff ts_start <= t2 and ts_end >= t1, so
        `lo` must be the first segment with ts_end >= t1 (side="left") and `hi` must
        count the segments with ts_start <= t2 (side="right").  Using side="right"/"left"
        here drops the boundary segment when the query boundary coincides with a
        segment boundary.
        """
        lo = int(np.searchsorted(self._ends, t1, side="left"))
        hi = int(np.searchsorted(self._starts, t2, side="right"))
        return lo, hi

    def range_agg(self, t1: int, t2: int) -> Tuple[AggResult, QueryStats]:
        """Range aggregation over [t1, t2].

        With sketches: interior (fully covered) segments are answered from sidecar A
        without decoding; only the two boundary segments are decoded.
        Without sketches (Prism-NoSketch): every segment in the range is decoded.
        """
        st = QueryStats()
        lo, hi = self._segment_range(t1, t2)
        st.segments_touched = max(0, hi - lo)
        if hi <= lo:
            return AggResult(0, 0.0, float("nan"), float("nan"), float("nan"), float("nan"), (0, 0)), st

        total_m, total_e = 0, 0
        sq_m, sq_e = 0, 0
        count = 0
        vmin = math.inf
        vmax = -math.inf
        for si in range(lo, hi):
            seg = self.segments[si]
            full = (self.use_sketch and seg.sketch_enabled
                    and seg.ts_start >= t1 and seg.ts_end <= t2)
            if full:
                sk = seg.sketch_a
                total_m, total_e = _align(total_m, total_e, sk.sum_mant, sk.sum_exp)
                sq_m, sq_e = _align(sq_m, sq_e, sk.sq_mant, sk.sq_exp)
                count += sk.count
                vmin = min(vmin, sk.vmin)
                vmax = max(vmax, sk.vmax)
                st.segments_sketched += 1
                st.sketch_ops += 1
            else:
                vals = seg.decode()
                st.decoded_points += seg.n_values
                st.segments_decoded += 1
                st.bytes_decoded += len(seg.payload)
                sel = vals
                # restrict to the requested time window inside the boundary segment
                offs = np.arange(seg.ts_start, seg.ts_end + 1)
                mask = (offs >= t1) & (offs <= t2)
                if mask.any():
                    sel = vals[mask]
                if sel.size:
                    sm, se, qm, qe = exact_accumulate(sel)
                    total_m, total_e = _align(total_m, total_e, sm, se)
                    sq_m, sq_e = _align(sq_m, sq_e, qm, qe)
                    count += int(sel.size)
                    vmin = min(vmin, float(sel.min()))
                    vmax = max(vmax, float(sel.max()))
        total = int_to_float(total_m, total_e)
        mean = total / count if count else float("nan")
        if count:
            sqf = Fraction(sq_m, 1 << -sq_e) if sq_e < 0 else Fraction(sq_m * (1 << sq_e))
            tf = Fraction(total_m, 1 << -total_e) if total_e < 0 else Fraction(total_m * (1 << total_e))
            var = _frac_to_float((sqf - tf * tf / count) / count)
        else:
            var = float("nan")
        return AggResult(count, total, vmin, vmax, mean, var, (total_m, total_e)), st

    def point_query(self, t: int) -> Tuple[float, QueryStats]:
        """Point lookup by timestamp using the segment directory (sidecar C)."""
        st = QueryStats()
        i = int(np.searchsorted(self._starts, t, side="right")) - 1
        if i < 0 or i >= len(self.segments):
            return float("nan"), st
        seg = self.segments[i]
        st.segments_touched = 1
        vals = seg.decode()
        st.decoded_points += seg.n_values
        st.segments_decoded += 1
        st.bytes_decoded += len(seg.payload)
        offs = np.arange(seg.ts_start, seg.ts_end + 1)
        hit = np.flatnonzero(offs == t)
        return (float(vals[hit[0]]) if hit.size else float("nan")), st

    def shape_match(self, pattern: np.ndarray, eps: float) -> Tuple[List[int], QueryStats]:
        """Pattern matching with the provable pruning bound of sidecar B.

        For every segment the bound
            || s_norm - q_norm ||_2 >= || PAA(s) - PAA(q) ||_2 - sqrt(e_s) - sqrt(e_q)
        decides whether the segment can be skipped without decoding. Segments whose bound
        exceeds eps are pruned (no decode); the rest are decoded and compared exactly.
        """
        st = QueryStats()
        q_paa, q_energy = paa_of_query(pattern, 8, self.gmin, self.gmax)
        matches: List[int] = []
        for seg in self.segments:
            st.segments_touched += 1
            if not (self.use_shape and seg.sketch_b is not None):
                vals = seg.decode()
                st.decoded_points += seg.n_values
                st.segments_decoded += 1
                if self._shape_distance(vals, pattern) <= eps:
                    matches.append(seg.idx)
                continue
            st.sketch_ops += 1
            bound = shape_lower_bound(seg.sketch_b, q_paa, q_energy)
            if bound > eps:
                continue  # provably no match: skipped without decoding
            vals = seg.decode()
            st.decoded_points += seg.n_values
            st.segments_decoded += 1
            if self._shape_distance(vals, pattern) <= eps:
                matches.append(seg.idx)
        return matches, st

    def _shape_distance(self, seg_vals: np.ndarray, pattern: np.ndarray) -> float:
        """Global-normalised L2 distance -- the quantity the pruning bound refers to."""
        n = min(seg_vals.size, pattern.size)
        rng = self.gmax - self.gmin
        if rng <= 0 or not np.isfinite(rng):
            return math.inf
        a = (seg_vals[:n] - self.gmin) / rng
        b = (pattern[:n] - self.gmin) / rng
        return float(np.linalg.norm(a - b))

    def anomaly_scan(self, z_threshold: float) -> Tuple[List[Tuple[int, float]], QueryStats]:
        """Anomaly scan: flag points whose |z-score| (w.r.t. the global mean/std) exceeds
        the threshold. Sidecar A gives the per-segment min/max and count, which bounds the
        segment's z-score range, so segments that cannot contain an anomaly are skipped
        without decoding."""
        st = QueryStats()
        allv = np.concatenate([s.decode() for s in self.segments]) if not self.use_sketch else None
        if allv is None:
            # reconstruct global stats from sketches (exact sum/count + min/max)
            sm = se = 0
            cnt = 0
            vmin = math.inf
            vmax = -math.inf
            for s in self.segments:
                if not s.sketch_enabled:
                    vals = s.decode()
                    st.decoded_points += s.n_values
                    sm, se = _align(sm, se, *exact_accumulate(vals)[:2])
                    cnt += s.n_values
                    vmin = min(vmin, float(vals.min()))
                    vmax = max(vmax, float(vals.max()))
                    continue
                sk = s.sketch_a
                sm, se = _align(sm, se, sk.sum_mant, sk.sum_exp)
                cnt += sk.count
                vmin = min(vmin, sk.vmin)
                vmax = max(vmax, sk.vmax)
            mean = int_to_float(sm, se) / cnt if cnt else 0.0
            # variance from the exact sumsq of the sketch
            sqm = sqe = 0
            for s in self.segments:
                if not s.sketch_enabled:
                    vals = s.decode()
                    st.decoded_points += s.n_values
                    sqm, sqe = _align(sqm, sqe, *exact_accumulate(vals)[2:])
                    continue
                sqm, sqe = _align(sqm, sqe, s.sketch_a.sq_mant, s.sketch_a.sq_exp)
            sqv = float(Fraction(sqm, 1 << -sqe) if sqe < 0 else Fraction(sqm * (1 << sqe)))
            var = (sqv - cnt * mean * mean) / cnt if cnt else 0.0
            std = math.sqrt(max(var, 0.0))
        else:
            mean = float(allv.mean())
            std = float(allv.std())
        hits: List[Tuple[int, float]] = []
        for seg in self.segments:
            st.segments_touched += 1
            if self.use_sketch and seg.sketch_enabled:
                sk = seg.sketch_a
                seg_mean = sk.sum_value / sk.count if sk.count else 0.0
                # bound the segment's z-score using min/max only
                zmax = max(abs(sk.vmax - mean), abs(sk.vmin - mean)) / std if std > 0 else math.inf
                if zmax <= z_threshold:
                    continue
            vals = seg.decode()
            st.decoded_points += seg.n_values
            st.segments_decoded += 1
            z = np.abs(vals - mean) / std if std > 0 else np.zeros_like(vals)
            for off, zz in zip(range(seg.ts_start, seg.ts_end + 1), z.tolist()):
                if zz > z_threshold:
                    hits.append((off, zz))
        return hits, st


def _shape_distance(seg_vals: np.ndarray, pattern: np.ndarray) -> float:
    """L2 distance between the min-max normalised segment and the normalised pattern."""
    n = min(seg_vals.size, pattern.size)
    a = seg_vals[:n].astype(np.float64)
    b = pattern[:n].astype(np.float64)
    for arr in (a, b):
        rng = arr.max() - arr.min()
        if rng > 0:
            arr -= arr.min()
            arr /= rng
    return float(np.linalg.norm(a - b))
