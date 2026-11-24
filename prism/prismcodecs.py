"""Python ports of the Chimp128, Gorilla, run-length/dictionary and plain codecs.

The Chimp128 and Gorilla streams are byte-identical to the reference Java
implementation of Chimp (gr.aueb.delorean.chimp.ChimpN with N=128, and
fi.iki.yak.ts.compression.gorilla.Compressor).  Bit order: MSB-first.
"""
from __future__ import annotations

import struct
from typing import List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# bit IO (MSB-first, big endian)
# --------------------------------------------------------------------------------------


class BitWriter:
    __slots__ = ("_buf", "_cur", "_n")

    def __init__(self) -> None:
        self._buf = bytearray()
        self._cur = 0
        self._n = 0

    def write_bit(self, b: int) -> None:
        self._cur = (self._cur << 1) | (b & 1)
        self._n += 1
        if self._n == 8:
            self._buf.append(self._cur)
            self._cur = 0
            self._n = 0

    def write_bits(self, value: int, nbits: int) -> None:
        """Write the low `nbits` bits of `value`, most significant first.

        Matches OutputBitStream.writeInt(x, n) / writeLong(x, n) and
        ByteBufferBitOutput.writeBits(value, n) of the official artifact.
        """
        if nbits == 0:
            return
        value &= (1 << nbits) - 1
        for i in range(nbits - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    def bit_length(self) -> int:
        return len(self._buf) * 8 + self._n

    def bytes_(self) -> bytes:
        out = bytearray(self._buf)
        if self._n:
            out.append(self._cur << (8 - self._n))
        return bytes(out)


class BitReader:
    __slots__ = ("_data", "_pos")

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

    def read_long(self, nbits: int) -> int:
        return self.read_bits(nbits)

    def bit_position(self) -> int:
        return self._pos


# --------------------------------------------------------------------------------------
# helpers shared by the official tables
# --------------------------------------------------------------------------------------

# gr.aueb.delorean.chimp.Chimp / ChimpN: leadingRepresentation (64 entries)
_LEADING_REPRESENTATION = (
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
)

# gr.aueb.delorean.chimp.Chimp / ChimpN: leadingRound (64 entries)
_LEADING_ROUND = (
    0, 0, 0, 0, 0, 0, 0, 0,
    8, 8, 8, 8, 12, 12, 12, 12,
    16, 16, 18, 18, 20, 20, 22, 22,
    24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24,
)

# ChimpNDecompressor.leadingRepresentation (8 entries, indexed by the 3-bit code)
_LEADING_FROM_CODE = (0, 8, 12, 16, 18, 20, 22, 24)

_NAN_LONG = 0x7FF8000000000000


def _nlz(x: int) -> int:
    """Java Long.numberOfLeadingZeros for a 64-bit value."""
    if x == 0:
        return 64
    return 64 - x.bit_length()


def _ntz(x: int) -> int:
    """Java Long.numberOfTrailingZeros for a 64-bit value."""
    if x == 0:
        return 64
    return (x & -x).bit_length() - 1


def _to_long(v: float) -> int:
    """Double.doubleToRawLongBits, returned as an unsigned 64-bit int."""
    return struct.unpack("<Q", struct.pack("<d", float(v)))[0]


def _from_long(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u & 0xFFFFFFFFFFFFFFFF))[0]


# --------------------------------------------------------------------------------------
# Chimp128  (official gr.aueb.delorean.chimp.ChimpN with previousValues = 128)
# --------------------------------------------------------------------------------------


class Chimp128Codec:
    """Bit-exact port of the official ChimpN(128) (the 'Chimp128' variant).

    Encoder state machine copied from ChimpN.java; the decoder follows
    ChimpNDecompressor.java.  `size` is the official bit counter (getSize()).
    """

    name = "chimp128"

    def __init__(self, previous_values: int = 128) -> None:
        self.previous_values = previous_values
        self.previous_values_log2 = int(np.log(previous_values) / np.log(2))
        self.threshold = 6 + self.previous_values_log2
        self.set_lsb = (1 << (self.threshold + 1)) - 1
        self.indices = [0] * (1 << (self.threshold + 1))
        self.stored_values = [0] * previous_values
        self.flag_zero_size = self.previous_values_log2 + 2
        self.flag_one_size = self.previous_values_log2 + 11
        self.reset()

    def reset(self) -> None:
        self._w = BitWriter()
        self._size = 0
        self._first = True
        self._stored_leading_zeros = 0x7FFFFFFF
        self._current = 0
        self._index = 0
        self.indices = [0] * len(self.indices)

    # -- encoder ------------------------------------------------------------------
    def _write_first(self, value: int) -> None:
        self._first = False
        self.stored_values[self._current] = value
        self._w.write_bits(value, 64)
        self.indices[(value & 0xFFFFFFFF) & self.set_lsb] = self._index
        self._size += 64

    def _compress_value(self, value: int) -> None:
        key = (value & 0xFFFFFFFF) & self.set_lsb
        curr_index = self.indices[key]
        trailing_zeros = 0
        if (self._index - curr_index) < self.previous_values:
            temp_xor = value ^ self.stored_values[curr_index % self.previous_values]
            trailing_zeros = _ntz(temp_xor)
            if trailing_zeros > self.threshold:
                previous_index = curr_index % self.previous_values
                xor = temp_xor
            else:
                previous_index = self._index % self.previous_values
                xor = self.stored_values[previous_index] ^ value
        else:
            previous_index = self._index % self.previous_values
            xor = self.stored_values[previous_index] ^ value

        if xor == 0:
            self._w.write_bits(previous_index, self.flag_zero_size)
            self._size += self.flag_zero_size
            self._stored_leading_zeros = 65
        else:
            leading_zeros = _LEADING_ROUND[_nlz(xor)]
            if trailing_zeros > self.threshold:
                significant_bits = 64 - leading_zeros - trailing_zeros
                self._w.write_bits(
                    512 * (self.previous_values + previous_index)
                    + 64 * _LEADING_REPRESENTATION[leading_zeros]
                    + significant_bits,
                    self.flag_one_size,
                )
                self._w.write_bits((xor >> trailing_zeros) & 0xFFFFFFFFFFFFFFFF, significant_bits)
                self._size += significant_bits + self.flag_one_size
                self._stored_leading_zeros = 65
            elif leading_zeros == self._stored_leading_zeros:
                significant_bits = 64 - leading_zeros
                self._w.write_bits(2, 2)
                self._w.write_bits(xor & 0xFFFFFFFFFFFFFFFF, significant_bits)
                self._size += 2 + significant_bits
            else:
                self._stored_leading_zeros = leading_zeros
                significant_bits = 64 - leading_zeros
                self._w.write_bits(24 + _LEADING_REPRESENTATION[leading_zeros], 5)
                self._w.write_bits(xor & 0xFFFFFFFFFFFFFFFF, significant_bits)
                self._size += 5 + significant_bits

        self._current = (self._current + 1) % self.previous_values
        self.stored_values[self._current] = value
        self._index += 1
        self.indices[key] = self._index

    def add_value(self, value: float) -> None:
        v = _to_long(value)
        if self._first:
            self._write_first(v)
        else:
            self._compress_value(v)

    def close(self) -> None:
        self.add_value(float("nan"))
        self._w.write_bit(0)

    @property
    def size_bits(self) -> int:
        return self._size

    def bytes_(self) -> bytes:
        return self._w.bytes_()

    # -- decoder ------------------------------------------------------------------
    @staticmethod
    def decode(data: bytes, previous_values: int = 128) -> np.ndarray:
        log2 = int(np.log(previous_values) / np.log(2))
        initial_fill = log2 + 9
        r = BitReader(data)
        stored_values = [0] * previous_values
        current = 0
        stored_leading_zeros = 0x7FFFFFFF
        stored_val = r.read_bits(64)
        stored_values[current] = stored_val
        if stored_val == _NAN_LONG:
            return np.empty(0, dtype=np.float64)
        out: List[float] = [_from_long(stored_val)]
        while True:
            flag = r.read_bits(2)
            if flag == 3:
                stored_leading_zeros = _LEADING_FROM_CODE[r.read_bits(3)]
                value = r.read_bits(64 - stored_leading_zeros)
                stored_val = stored_val ^ value
            elif flag == 2:
                value = r.read_bits(64 - stored_leading_zeros)
                stored_val = stored_val ^ value
            elif flag == 1:
                fill = initial_fill
                temp = r.read_bits(fill)
                index = (temp >> (fill - log2)) & ((1 << log2) - 1)
                fill -= log2
                stored_leading_zeros = _LEADING_FROM_CODE[(temp >> (fill - 3)) & 7]
                fill -= 3
                significant_bits = (temp >> (fill - 6)) & 63
                fill -= 6
                if significant_bits == 0:
                    significant_bits = 64
                stored_trailing_zeros = 64 - significant_bits - stored_leading_zeros
                stored_val = stored_values[index]
                value = r.read_bits(64 - stored_leading_zeros - stored_trailing_zeros)
                value = (value << stored_trailing_zeros) & 0xFFFFFFFFFFFFFFFF
                stored_val = stored_val ^ value
            else:
                stored_val = stored_values[r.read_bits(log2)]
            if stored_val == _NAN_LONG:
                break
            out.append(_from_long(stored_val))
            current = (current + 1) % previous_values
            stored_values[current] = stored_val
        return np.asarray(out, dtype=np.float64)

    # -- convenience ---------------------------------------------------------------
    @classmethod
    def encode(cls, values: Sequence[float], previous_values: int = 128) -> bytes:
        c = cls(previous_values)
        for v in values:
            c.add_value(v)
        c.close()
        return c.bytes_()


# --------------------------------------------------------------------------------------
# Gorilla (official fi.iki.yak.ts.compression.gorilla.Compressor, value stream only)
# --------------------------------------------------------------------------------------


class GorillaCodec:
    """Bit-exact port of the official 'Gorilla' value compressor.

    Note this is the *modified* Gorilla of the artifact (leading/trailing zeros are
    remembered and reused), not the original Facebook paper format.
    """

    name = "gorilla"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._w = BitWriter()
        self._size = 0
        self._first = True
        self._stored_leading_zeros = 0x7FFFFFFF
        self._stored_trailing_zeros = 0
        self._stored_val = 0

    def add_value(self, value: float) -> None:
        v = _to_long(value)
        if self._first:
            self._first = False
            self._stored_val = v
            self._w.write_bits(v, 64)
            self._size += 64
            return
        xor = self._stored_val ^ v
        if xor == 0:
            self._w.write_bit(0)
            self._size += 1
        else:
            leading_zeros = _nlz(xor)
            trailing_zeros = _ntz(xor)
            if leading_zeros >= 32:
                leading_zeros = 31
            self._w.write_bit(1)
            self._size += 1
            if (leading_zeros >= self._stored_leading_zeros
                    and trailing_zeros >= self._stored_trailing_zeros):
                significant_bits = 64 - self._stored_leading_zeros - self._stored_trailing_zeros
                self._w.write_bit(0)
                self._w.write_bits(
                    (xor >> self._stored_trailing_zeros) & 0xFFFFFFFFFFFFFFFF, significant_bits
                )
                self._size += 1 + significant_bits
            else:
                self._w.write_bit(1)
                self._w.write_bits(leading_zeros, 5)
                significant_bits = 64 - leading_zeros - trailing_zeros
                if significant_bits == 64:
                    self._w.write_bits(0, 6)
                else:
                    self._w.write_bits(significant_bits, 6)
                self._w.write_bits(
                    (xor >> trailing_zeros) & 0xFFFFFFFFFFFFFFFF, significant_bits
                )
                self._stored_leading_zeros = leading_zeros
                self._stored_trailing_zeros = trailing_zeros
                self._size += 1 + 5 + 6 + significant_bits
        self._stored_val = v

    def close(self) -> None:
        self.add_value(float("nan"))
        self._w.write_bit(0)

    @property
    def size_bits(self) -> int:
        return self._size

    def bytes_(self) -> bytes:
        return self._w.bytes_()

    @staticmethod
    def decode(data: bytes) -> np.ndarray:
        """Mirror of the official fi.iki.yak.ts.compression.gorilla.Decompressor.

        Stream grammar (per value):
          control bit 0            -> value repeats (xor == 0)
          control bit 1, next 0    -> reuse the stored leading/trailing zero counts
          control bit 1, next 1    -> 5 bits leading zeros, 6 bits significant
                                     length (0 encodes 64)
        """
        r = BitReader(data)
        stored_leading_zeros = 0x7FFFFFFF
        stored_trailing_zeros = 0
        stored_val = r.read_bits(64)
        if stored_val == _NAN_LONG:
            return np.empty(0, dtype=np.float64)
        out: List[float] = [_from_long(stored_val)]
        while True:
            if r.read_bit() == 0:
                cur = stored_val
            else:
                if r.read_bit() == 1:
                    stored_leading_zeros = r.read_bits(5)
                    significant = r.read_bits(6)
                    if significant == 0:
                        significant = 64
                    stored_trailing_zeros = 64 - significant - stored_leading_zeros
                value = r.read_bits(64 - stored_leading_zeros - stored_trailing_zeros)
                value = (value << stored_trailing_zeros) & 0xFFFFFFFFFFFFFFFF
                cur = stored_val ^ value
            if cur == _NAN_LONG:
                break
            out.append(_from_long(cur))
            stored_val = cur
        return np.asarray(out, dtype=np.float64)

    @classmethod
    def encode(cls, values: Sequence[float]) -> bytes:
        c = cls()
        for v in values:
            c.add_value(v)
        c.close()
        return c.bytes_()


# --------------------------------------------------------------------------------------
# uncompressed / dictionary baselines
# --------------------------------------------------------------------------------------


class PlainCodec:
    """Uncompressed float64 payload."""

    name = "plain"

    @staticmethod
    def encode(values: Sequence[float]) -> bytes:
        return np.asarray(values, dtype=np.float64).tobytes()

    @staticmethod
    def decode(data: bytes, n: int) -> np.ndarray:
        return np.frombuffer(data[: 8 * n], dtype=np.float64).copy()


class RLEDictCodec:
    """Dictionary + run-length codec (IoTDB-style light-weight encoding baseline).

    Format: u32 distinct-count, u32*distinct (raw f64), u32 run-count,
    then per run (u32 value-index, u32 run-length).
    """

    name = "rledict"

    @staticmethod
    def encode(values: Sequence[float]) -> bytes:
        vals = np.asarray(values, dtype=np.float64)
        ids: dict = {}
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
        (ndistinct,) = struct.unpack_from("<I", data, off)
        off += 4
        distinct = np.empty(ndistinct, dtype=np.float64)
        for i in range(ndistinct):
            distinct[i] = struct.unpack_from("<d", data, off)[0]
            off += 8
        (nruns,) = struct.unpack_from("<I", data, off)
        off += 4
        out = np.empty(n, dtype=np.float64)
        pos = 0
        for _ in range(nruns):
            vi, rl = struct.unpack_from("<II", data, off)
            off += 8
            out[pos: pos + rl] = distinct[vi]
            pos += rl
        return out[:n]


CODECS = {
    "plain": PlainCodec,
    "gorilla": GorillaCodec,
    "chimp128": Chimp128Codec,
    "rledict": RLEDictCodec,
}
