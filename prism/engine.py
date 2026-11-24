"""Storage formats and query executor.

Formats
  Uncompressed          raw float64 columns; reference answers.
  IoTDB                 TsFile layout: run-length encoded pages, LZ4 per chunk; a query
                        decompresses every chunk it touches.
  CompressIoTDB-NoLate  CompressIoTDB operators on run-length runs with eager LZ4
                        decompression of the chunk.
  CompressIoTDB         the same operators with page-level late decompression.
  Prism                 independently decodable segments with a SegmentSketch, a
                        ShapeIndex and a segment directory, executed sidecar-first.

Every operator returns QStats:
  bytes_read      compressed bytes fetched from storage
  decoded_points  float64 values materialized
  items_scanned   encoded items processed (values, runs, segments, sidecar records)
"""
from __future__ import annotations

import math
import os
import struct
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import lz4.block
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fastcodecs  # noqa: E402
from core import (  # noqa: E402
    SidecarA,
    SidecarB,
    _align,
    _frac_to_float,
    exact_accumulate,
    int_to_float,
    shape_prune_rates as core_shape_prune_rates,
)
# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------

S_DEFAULT = 512          # segment size
CHUNK = 1024             # TsFile chunk size in values
PAGE = 256               # values per page inside a chunk
SHAPE_K = 32             # PAA blocks per segment for the ShapeIndex


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------


@dataclass
class QStats:
    latency_ms: float = 0.0
    bytes_read: int = 0
    decoded_points: int = 0
    items_scanned: int = 0
    segments_touched: int = 0
    sketch_hits: int = 0
    sketch_reads: int = 0

    def __add__(self, o: "QStats") -> "QStats":
        return QStats(
            self.latency_ms + o.latency_ms,
            self.bytes_read + o.bytes_read,
            self.decoded_points + o.decoded_points,
            self.items_scanned + o.items_scanned,
            self.segments_touched + o.segments_touched,
            self.sketch_hits + o.sketch_hits,
            self.sketch_reads + o.sketch_reads,
        )

    def __iadd__(self, o: "QStats") -> "QStats":
        self.latency_ms += o.latency_ms
        self.bytes_read += o.bytes_read
        self.decoded_points += o.decoded_points
        self.items_scanned += o.items_scanned
        self.segments_touched += o.segments_touched
        self.sketch_hits += o.sketch_hits
        self.sketch_reads += o.sketch_reads
        return self


@dataclass
class Agg:
    """Aggregate result (count / total / min / max / sumsq).

    `var_override` carries a variance that was computed exactly (Prism merges the
    sidecars' exact rational sums and derives the variance from them).  The float
    `sumsq` slot is then only used for the multi-column merge path, where the
    variance is re-derived from it exactly as in the other stores.
    """

    count: int = 0
    total: float = 0.0
    vmin: float = math.inf
    vmax: float = -math.inf
    sumsq: float = 0.0
    var_override: Optional[float] = None

    def __add__(self, o: "Agg") -> "Agg":
        return Agg(self.count + o.count, self.total + o.total,
                   min(self.vmin, o.vmin), max(self.vmax, o.vmax),
                   self.sumsq + o.sumsq, None)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else float("nan")

    @property
    def var(self) -> float:
        if self.var_override is not None:
            return self.var_override
        if self.count == 0:
            return float("nan")
        return (self.sumsq - self.total * self.total / self.count) / self.count

    def digest(self) -> Tuple[int, float, float]:
        return (self.count, round(self.total, 6), round(self.var, 6))


# --------------------------------------------------------------------------------------
# query specifications
# --------------------------------------------------------------------------------------


@dataclass
class QuerySpec:
    """A query instantiated on one dataset.

    kind:
      range_agg      aggregate over [lo, hi) of one or more columns
      filter_expr    value filter + affine expression + aggregate
      window_agg     tumbling-window aggregate over [lo, hi)
      point          point lookup
      match          shape query over segment-aligned windows
      anom           anomaly scan (|z| > threshold)
    """

    qid: str
    kind: str
    cols: Tuple[int, ...] = ()
    lo: int = 0
    hi: int = 0
    window: int = 0
    thr: float = 0.0
    expr_a: float = 1.0
    expr_b: float = 0.0
    eps: float = 0.0
    zthr: float = 3.0
    paper_sql: str = ""
    paper_dataset: str = ""


# --------------------------------------------------------------------------------------
# base store
# --------------------------------------------------------------------------------------


class BaseStore:
    name = "base"

    def __init__(self, n: int) -> None:
        self.n = n
        self.stored_bytes = 0

    # -- interface ---------------------------------------------------------------------
    def agg_range(self, col: int, lo: int, hi: int) -> Tuple[Agg, QStats]:
        raise NotImplementedError

    def filter_expr(self, col: int, lo: int, hi: int, thr: float,
                    a: float, b: float) -> Tuple[Agg, QStats]:
        raise NotImplementedError

    def window_agg(self, col: int, lo: int, hi: int, window: int) -> Tuple[List[Agg], QStats]:
        raise NotImplementedError

    def filter_window_agg(self, col: int, lo: int, hi: int, window: int, thr: float,
                          a: float, b: float) -> Tuple[List[Agg], QStats]:
        """Value filter + tumbling-window aggregate (Q7 operator class:
        "SELECT VARIANCE(direction) FROM root.lr WHERE lane = 0 GROUP BY (..., 1d)")."""
        raise NotImplementedError

    def point(self, col: int, idx: int) -> Tuple[float, QStats]:
        raise NotImplementedError

    def match(self, col: int, pattern: np.ndarray, eps: float) -> Tuple[List[int], QStats]:
        raise NotImplementedError

    def anom(self, col: int, zthr: float) -> Tuple[List[int], QStats]:
        raise NotImplementedError
# --------------------------------------------------------------------------------------
# Uncompressed
# --------------------------------------------------------------------------------------


class UncompressedStore(BaseStore):
    name = "Uncompressed"

    def __init__(self, cols: np.ndarray) -> None:
        super().__init__(cols.shape[0])
        self.cols = cols
        self.stored_bytes = cols.size * 8

    def agg_range(self, col: int, lo: int, hi: int) -> Tuple[Agg, QStats]:
        arr = self.cols[lo:hi, col]
        st = QStats(bytes_read=arr.size * 8, decoded_points=arr.size,
                   items_scanned=arr.size)
        if arr.size == 0:
            return Agg(), st
        a = arr.astype(np.float64)
        return Agg(arr.size, float(a.sum()), float(a.min()), float(a.max()),
                   float((a ** 2).sum())), st

    def filter_expr(self, col: int, lo: int, hi: int, thr: float,
                    a: float, b: float) -> Tuple[Agg, QStats]:
        arr = self.cols[lo:hi, col]
        st = QStats(bytes_read=arr.size * 8, decoded_points=arr.size,
                   items_scanned=arr.size)
        sel = arr[arr > thr]
        if sel.size == 0:
            return Agg(), st
        e = a * sel + b
        return Agg(sel.size, float(e.sum()), float(e.min()), float(e.max()),
                   float((e ** 2).sum())), st

    def window_agg(self, col: int, lo: int, hi: int, window: int) -> Tuple[List[Agg], QStats]:
        arr = self.cols[lo:hi, col].astype(np.float64)
        st = QStats(bytes_read=arr.size * 8, decoded_points=arr.size,
                    items_scanned=arr.size)
        if arr.size == 0 or window <= 0:
            return [], st
        nw = (arr.size + window - 1) // window
        out: List[Agg] = []
        for k in range(nw):
            w = arr[k * window:(k + 1) * window]
            if w.size == 0:
                continue
            out.append(Agg(w.size, float(w.sum()), float(w.min()), float(w.max()),
                           float((w ** 2).sum())))
        return out, st

    def filter_window_agg(self, col: int, lo: int, hi: int, window: int, thr: float,
                          a: float, b: float) -> Tuple[List[Agg], QStats]:
        """Value filter + tumbling-window aggregate (Q7 operator class).

        The filter is applied to the values of the window; the aggregate is over the
        surviving values of that window.
        """
        arr = self.cols[lo:hi, col].astype(np.float64)
        st = QStats(bytes_read=arr.size * 8, decoded_points=arr.size,
                    items_scanned=arr.size)
        out: List[Agg] = []
        if window <= 0 or hi <= lo:
            return out, st
        nw = (arr.size + window - 1) // window
        for k in range(nw):
            w = arr[k * window:(k + 1) * window]
            w = w[w > thr]
            if w.size == 0:
                continue
            e = a * w + b
            out.append(Agg(w.size, float(e.sum()), float(e.min()), float(e.max()),
                           float((e ** 2).sum())))
        return out, st

    def point(self, col: int, idx: int) -> Tuple[float, QStats]:
        st = QStats(bytes_read=8, decoded_points=1, items_scanned=1)
        return float(self.cols[idx, col]), st

    def match(self, col: int, pattern: np.ndarray, eps: float) -> Tuple[List[int], QStats]:
        arr = self.cols[:, col].astype(np.float64)
        st = QStats(bytes_read=arr.size * 8, decoded_points=arr.size,
                   items_scanned=arr.size)
        return _brute_match(arr, pattern, eps), st

    def anom(self, col: int, zthr: float) -> Tuple[List[int], QStats]:
        arr = self.cols[:, col].astype(np.float64)
        st = QStats(bytes_read=arr.size * 8, decoded_points=arr.size,
                   items_scanned=arr.size)
        mu = float(arr.mean())
        sd = float(arr.std())
        if sd <= 0:
            return [], st
        z = np.abs(arr - mu) / sd
        return np.flatnonzero(z > zthr).tolist(), st


def _brute_match(arr: np.ndarray, pattern: np.ndarray, eps: float) -> List[int]:
    """Segment-aligned shape search on globally min-max normalised values."""
    n = arr.size
    L = pattern.size
    if L == 0 or L > n:
        return []
    gmin, gmax = float(arr.min()), float(arr.max())
    rng = gmax - gmin
    if rng <= 0:
        return []
    a = (arr - gmin) / rng
    b = (pattern - gmin) / rng
    hits: List[int] = []
    for s in range(0, n - L + 1, L):
        d = float(np.linalg.norm(a[s:s + L] - b) / math.sqrt(L))
        if d <= eps:
            hits.append(s)
    return hits


# --------------------------------------------------------------------------------------
# RLE light-weight encoding (the layer used by IoTDB and CompressIoTDB)
# --------------------------------------------------------------------------------------


class RLEColumn:
    """Run-length encoded float64 column: (value, run-length) pairs.

    The on-disk record is 12 bytes: <f8 value><i4 count>, exactly the layout of
    IoTDB's light-weight RLE encoding.  Serialisation is vectorised (a numpy
    structured dtype) because it sits in the hot path of every baseline query.
    """

    __slots__ = ("values", "counts")

    _DT = np.dtype({"names": ["v", "c"], "formats": ["<f8", "<i4"],
                    "offsets": [0, 8], "itemsize": 12})

    def __init__(self, values: np.ndarray, counts: np.ndarray) -> None:
        self.values = values
        self.counts = counts

    @staticmethod
    def build(arr: np.ndarray) -> "RLEColumn":
        if arr.size == 0:
            return RLEColumn(np.zeros(0, np.float64), np.zeros(0, np.int64))
        canon = arr.astype(np.float64, copy=True)
        canon[canon == 0.0] = 0.0
        change = np.flatnonzero(np.diff(canon) != 0.0) + 1
        starts = np.concatenate(([0], change))
        counts = np.diff(np.concatenate((starts, [canon.size]))).astype(np.int64)
        return RLEColumn(canon[starts].copy(), counts)

    def serialise(self) -> bytes:
        rec = np.empty(self.values.size, dtype=self._DT)
        rec["v"] = self.values
        rec["c"] = self.counts
        return rec.tobytes()

    @staticmethod
    def deserialise(buf: bytes) -> "RLEColumn":
        rec = np.frombuffer(buf, dtype=RLEColumn._DT)
        return RLEColumn(rec["v"].astype(np.float64), rec["c"].astype(np.int64))

    def expand(self) -> np.ndarray:
        return np.repeat(self.values, self.counts)


def _rle_page(arr: np.ndarray) -> Tuple[RLEColumn, bytes]:
    """One page: light-weight RLE runs, then the LZ4 layer."""
    col = RLEColumn.build(arr)
    return col, lz4.block.compress(col.serialise())


# --------------------------------------------------------------------------------------
# IoTDB / CompressIoTDB: chunked, page-level RLE + LZ4
# --------------------------------------------------------------------------------------


class ChunkedStore(BaseStore):
    """Shared layout: values -> pages of PAGE values -> RLE -> LZ4, grouped in chunks.

    `late` selects between the two CompressIoTDB variants:
      late=False  the whole chunk's LZ4 layer is decompressed eagerly (NoLate)
      late=True   only the pages the scan touches are decompressed (late decompression)
    The IoTDB baseline always decompresses the whole chunk and materialises every value.
    """

    def __init__(self, cols: np.ndarray, name: str, late: bool,
                 homomorphic: bool) -> None:
        super().__init__(cols.shape[0])
        self.cols = cols
        self.name = name
        self.late = late
        self.homomorphic = homomorphic
        # The page index is GLOBAL: page p holds the values [p*PAGE, (p+1)*PAGE).
        # (A local page index inside the chunk would alias the pages of the
        # different chunks and silently drop data.)
        self.pages: Dict[Tuple[int, int], int] = {}
        self.raw: Dict[Tuple[int, int], bytes] = {}
        for c in range(cols.shape[1]):
            for p in range(0, (self.n + PAGE - 1) // PAGE):
                s = p * PAGE
                e = min(s + PAGE, self.n)
                _, comp = _rle_page(cols[s:e, c])
                self.pages[(c, p)] = len(comp)
                self.raw[(c, p)] = comp
        self.chunk_pages = CHUNK // PAGE
        self.stored_bytes = sum(self.pages.values())

    # -- page helpers ------------------------------------------------------------------
    def _page_index(self, idx: int) -> int:
        return idx // PAGE

    def _chunk_of_page(self, page_idx: int) -> int:
        return page_idx // self.chunk_pages

    def _pages_of_chunk(self, ci: int) -> List[int]:
        base = ci * self.chunk_pages
        return [p for p in range(base, base + self.chunk_pages)
                if (0, p) in self.pages]

    def _load_page(self, col: int, page_idx: int) -> Tuple[RLEColumn, int]:
        """Decompress one page; returns (runs, compressed bytes read)."""
        raw = self._raw_page(col, page_idx)
        return RLEColumn.deserialise(lz4.block.decompress(raw)), len(raw)

    def _raw_page(self, col: int, page_idx: int) -> bytes:
        return self.raw[(col, page_idx)]

    def _materialise(self, col: int, ci: int) -> Tuple[np.ndarray, int, int]:
        """Eagerly decompress a whole chunk -> (values, bytes read, values decoded).

        `values decoded` counts the ORIGINAL float64 values that are materialised
        (c.counts.sum()), not the number of RLE runs: the run count is the encoded
        item count and is reported separately through items_scanned.
        """
        vals = []
        total_bytes = 0
        total_vals = 0
        for p in self._pages_of_chunk(ci):
            raw = self._raw_page(col, p)
            total_bytes += len(raw)
            c = RLEColumn.deserialise(lz4.block.decompress(raw))
            vals.append(c.expand())
            total_vals += int(c.counts.sum())
        return (np.concatenate(vals) if vals else np.zeros(0)), total_bytes, total_vals


class IoTDBStore(ChunkedStore):
    """Apache IoTDB / TsFile: RLE + LZ4 at chunk granularity, whole chunk decoded."""

    name = "IoTDB"

    def __init__(self, cols: np.ndarray) -> None:
        super().__init__(cols, "IoTDB", late=False, homomorphic=False)

    def _raw_page(self, col: int, page_idx: int) -> bytes:
        return self.raw[(col, page_idx)]
    def agg_range(self, col: int, lo: int, hi: int) -> Tuple[Agg, QStats]:
        st = QStats()
        if hi <= lo:
            return Agg(), st
        c0, c1 = lo // CHUNK, (hi - 1) // CHUNK
        parts = []
        for ci in range(c0, c1 + 1):
            vals, nb, nv = self._materialise(col, ci)
            st.bytes_read += nb
            st.decoded_points += nv
            st.items_scanned += nv
            st.segments_touched += 1
            parts.append(vals)
        arr = np.concatenate(parts)
        base = c0 * CHUNK
        sel = arr[lo - base:hi - base]
        return _agg_np(sel), st

    def filter_expr(self, col: int, lo: int, hi: int, thr: float,
                    a: float, b: float) -> Tuple[Agg, QStats]:
        st = QStats()
        c0, c1 = lo // CHUNK, (hi - 1) // CHUNK
        parts = []
        for ci in range(c0, c1 + 1):
            vals, nb, nv = self._materialise(col, ci)
            st.bytes_read += nb
            st.decoded_points += nv
            st.items_scanned += nv
            st.segments_touched += 1
            parts.append(vals)
        arr = np.concatenate(parts)
        base = c0 * CHUNK
        sel = arr[lo - base:hi - base]
        sel = sel[sel > thr]
        if sel.size == 0:
            return Agg(), st
        e = a * sel + b
        return Agg(sel.size, float(e.sum()), float(e.min()), float(e.max()),
                   float((e ** 2).sum())), st

    def filter_window_agg(self, col: int, lo: int, hi: int, window: int, thr: float,
                          a: float, b: float) -> Tuple[List[Agg], QStats]:
        """Filter + tumbling-window aggregate on the materialised chunk values."""
        st = QStats()
        if hi <= lo or window <= 0:
            return [], st
        c0, c1 = lo // CHUNK, (hi - 1) // CHUNK
        parts = []
        for ci in range(c0, c1 + 1):
            vals, nb, nv = self._materialise(col, ci)
            st.bytes_read += nb
            st.decoded_points += nv
            st.items_scanned += nv
            st.segments_touched += 1
            parts.append(vals)
        arr = np.concatenate(parts)
        base = c0 * CHUNK
        sel = arr[lo - base:hi - base]
        out: List[Agg] = []
        nw = (sel.size + window - 1) // window
        for k in range(nw):
            w = sel[k * window:(k + 1) * window]
            w = w[w > thr]
            if w.size == 0:
                continue
            e = a * w + b
            out.append(Agg(w.size, float(e.sum()), float(e.min()), float(e.max()),
                           float((e ** 2).sum())))
        return out, st

    def window_agg(self, col: int, lo: int, hi: int, window: int) -> Tuple[List[Agg], QStats]:
        st = QStats()
        if hi <= lo:
            return [], st
        c0, c1 = lo // CHUNK, (hi - 1) // CHUNK
        parts = []
        for ci in range(c0, c1 + 1):
            vals, nb, nv = self._materialise(col, ci)
            st.bytes_read += nb
            st.decoded_points += nv
            st.items_scanned += nv
            st.segments_touched += 1
            parts.append(vals)
        arr = np.concatenate(parts)
        base = c0 * CHUNK
        sel = arr[lo - base:hi - base]
        return _windows_np(sel, window), st

    def point(self, col: int, idx: int) -> Tuple[float, QStats]:
        ci = idx // CHUNK
        vals, nb, nv = self._materialise(col, ci)
        st = QStats(bytes_read=nb, decoded_points=nv, items_scanned=nv,
                   segments_touched=1)
        return float(vals[idx - ci * CHUNK]), st

    def match(self, col: int, pattern: np.ndarray, eps: float) -> Tuple[List[int], QStats]:
        st = QStats()
        vals = []
        for ci in range(self.n // CHUNK + 1):
            v, nb, nv = self._materialise(col, ci)
            if v.size == 0:
                continue
            st.bytes_read += nb
            st.decoded_points += nv
            st.items_scanned += nv
            st.segments_touched += 1
            vals.append(v)
        arr = np.concatenate(vals) if vals else np.zeros(0)
        return _brute_match(arr, pattern, eps), st

    def anom(self, col: int, zthr: float) -> Tuple[List[int], QStats]:
        st = QStats()
        vals = []
        for ci in range(self.n // CHUNK + 1):
            v, nb, nv = self._materialise(col, ci)
            if v.size == 0:
                continue
            st.bytes_read += nb
            st.decoded_points += nv
            st.items_scanned += nv
            st.segments_touched += 1
            vals.append(v)
        arr = np.concatenate(vals) if vals else np.zeros(0)
        mu = float(arr.mean())
        sd = float(arr.std())
        if sd <= 0:
            return [], st
        z = np.abs(arr - mu) / sd
        return np.flatnonzero(z > zthr).tolist(), st


def _run_positions(c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Start/end value positions of the RLE runs (the runs tile the range in order)."""
    ends = np.cumsum(c)
    return ends - c, ends


def _window_agg_runs(v: np.ndarray, c: np.ndarray, window: int,
                     keep: Optional[np.ndarray] = None,
                     a: float = 1.0, b: float = 0.0) -> List[Agg]:
    """Exact per-window aggregates over RLE runs.

    The runs tile the queried range in order, so a run's value positions are the
    cumulative sums of the run counts.  A run that straddles a window boundary
    contributes only the part inside the window: without this clipping the
    windowed aggregate would differ from the value-wise answer.
    """
    out: List[Agg] = []
    if v.size == 0 or window <= 0:
        return out
    starts, ends = _run_positions(c)
    n_vals = int(c.sum())
    for start in range(0, n_vals, window):
        end = min(start + window, n_vals)
        i0 = int(np.searchsorted(ends, start, side="right"))
        i1 = int(np.searchsorted(starts, end, side="left"))
        if i1 <= i0:
            continue
        left = np.maximum(starts[i0:i1], start)
        right = np.minimum(ends[i0:i1], end)
        cnt = (right - left).astype(np.float64)
        vals = v[i0:i1]
        if keep is not None:
            m = keep[i0:i1]
            cnt = cnt[m]
            vals = vals[m]
        if vals.size == 0:
            continue
        e = a * vals + b
        out.append(Agg(int(cnt.sum()), float((e * cnt).sum()),
                       float(e.min()), float(e.max()),
                       float((e * e * cnt).sum())))
    return out


class CompressIoTDBStore(ChunkedStore):
    """The CompColumn operators of CompressIoTDB (Tang et al., PVLDB 18(10), 2025).

    Operators are computed HOMOMORPHICALLY on the RLE runs: a run contributes
    (count, value) and the aggregate is accumulated run-wise, so the original values
    are never materialised (Sec. 4.3.1).  `late=True` defers the LZ4
    decompression to the pages the scan touches (Sec. 4.4.2).
    """

    def __init__(self, cols: np.ndarray, late: bool) -> None:
        super().__init__(cols, "CompressIoTDB" if late else "CompressIoTDB-NoLate",
                         late=late, homomorphic=True)

    def _raw_page(self, col: int, page_idx: int) -> bytes:
        return self.raw[(col, page_idx)]
    # -- page selection ----------------------------------------------------------------
    def _pages_for_range(self, col: int, lo: int, hi: int) -> List[int]:
        if hi <= lo:
            return []
        p0, p1 = lo // PAGE, (hi - 1) // PAGE
        return [p for p in range(p0, p1 + 1) if (col, p) in self.pages]

    def _runs_for_range(self, col: int, lo: int, hi: int) -> Tuple[np.ndarray, np.ndarray, QStats]:
        """Return the RLE runs overlapping [lo, hi), clipped to the range.

        The boundary runs are clipped so that the aggregate of a partial range is
        exact (the count of a clipped run is reduced, its value is unchanged).
        """
        st = QStats()
        if hi <= lo:
            return np.zeros(0), np.zeros(0, dtype=np.int64), st
        if self.late:
            pages = self._pages_for_range(col, lo, hi)
        else:
            # NoLate: the whole chunk's LZ4 layer is decompressed eagerly
            c0, c1 = lo // CHUNK, (hi - 1) // CHUNK
            pages = []
            for ci in range(c0, c1 + 1):
                pages.extend(self._pages_of_chunk(ci))
        vals_l, cnts_l = [], []
        for p in sorted(set(pages)):
            raw = self.raw[(col, p)]
            st.bytes_read += len(raw)
            col_rle = RLEColumn.deserialise(lz4.block.decompress(raw))
            base = p * PAGE
            starts = base + np.concatenate(([0], np.cumsum(col_rle.counts)[:-1]))
            ends = starts + col_rle.counts
            keep = (starts < hi) & (ends > lo)
            if not keep.any():
                continue
            st.items_scanned += int(col_rle.values.size)
            st.segments_touched += 1
            left = np.maximum(starts[keep], lo)
            right = np.minimum(ends[keep], hi)
            cnts = (right - left).astype(np.int64)
            m = cnts > 0
            if not m.any():
                continue
            vals_l.append(col_rle.values[keep][m])
            cnts_l.append(cnts[m])
        if not vals_l:
            return np.zeros(0), np.zeros(0, dtype=np.int64), st
        return np.concatenate(vals_l), np.concatenate(cnts_l), st

    def agg_range(self, col: int, lo: int, hi: int) -> Tuple[Agg, QStats]:
        v, c, st = self._runs_for_range(col, lo, hi)
        if v.size == 0:
            return Agg(), st
        return _agg_runs(v, c), st

    def _page_base_for(self, col: int, lo: int, hi: int, v, c) -> int:
        return 0

    def filter_expr(self, col: int, lo: int, hi: int, thr: float,
                    a: float, b: float) -> Tuple[Agg, QStats]:
        v, c, st = self._runs_for_range(col, lo, hi)
        if v.size == 0:
            return Agg(), st
        m = v > thr
        if not m.any():
            return Agg(), st
        e = a * v[m] + b
        cc = c[m].astype(np.float64)
        return Agg(int(cc.sum()), float((e * cc).sum()), float(e.min()), float(e.max()),
                   float((e ** 2 * cc).sum())), st

    def window_agg(self, col: int, lo: int, hi: int, window: int) -> Tuple[List[Agg], QStats]:
        v, c, st = self._runs_for_range(col, lo, hi)
        return _window_agg_runs(v, c, window), st

    def point(self, col: int, idx: int) -> Tuple[float, QStats]:
        p = idx // PAGE
        raw = self.raw[(col, p)]
        col_rle = RLEColumn.deserialise(lz4.block.decompress(raw))
        st = QStats(bytes_read=len(raw), items_scanned=int(col_rle.values.size),
                   segments_touched=1)
        run_start = np.concatenate(([0], np.cumsum(col_rle.counts)[:-1]))
        off = idx - p * PAGE
        r = int(np.searchsorted(run_start, off, side="right") - 1)
        return float(col_rle.values[r]), st

    def filter_window_agg(self, col: int, lo: int, hi: int, window: int, thr: float,
                          a: float, b: float) -> Tuple[List[Agg], QStats]:
        """Value filter + tumbling-window aggregate, computed on the RLE runs.

        A run carries a single value, so the filter is a run-level predicate and
        the surviving runs are clipped to the window bounds exactly.
        """
        v, c, st = self._runs_for_range(col, lo, hi)
        keep = (v > thr) if v.size else np.zeros(0, dtype=bool)
        return _window_agg_runs(v, c, window, keep=keep, a=a, b=b), st

    def match(self, col: int, pattern: np.ndarray, eps: float) -> Tuple[List[int], QStats]:
        """Shape query: the pages are materialised and scanned."""
        st = QStats()
        vals = []
        for p in range(self.n // PAGE + 1):
            if (col, p) not in self.pages:
                continue
            raw = self.raw[(col, p)]
            st.bytes_read += len(raw)
            col_rle = RLEColumn.deserialise(lz4.block.decompress(raw))
            vals.append(col_rle.expand())
            st.decoded_points += int(col_rle.counts.sum())
            st.items_scanned += int(col_rle.values.size)
            st.segments_touched += 1
        arr = np.concatenate(vals) if vals else np.zeros(0)
        return _brute_match(arr, pattern, eps), st

    def anom(self, col: int, zthr: float) -> Tuple[List[int], QStats]:
        """Anomaly scan: the pages are materialised and scanned."""
        st = QStats()
        vals = []
        for p in range(self.n // PAGE + 1):
            if (col, p) not in self.pages:
                continue
            raw = self.raw[(col, p)]
            st.bytes_read += len(raw)
            col_rle = RLEColumn.deserialise(lz4.block.decompress(raw))
            vals.append(col_rle.expand())
            st.decoded_points += int(col_rle.counts.sum())
            st.items_scanned += int(col_rle.values.size)
            st.segments_touched += 1
        arr = np.concatenate(vals) if vals else np.zeros(0)
        mu = float(arr.mean())
        sd = float(arr.std())
        if sd <= 0:
            return [], st
        z = np.abs(arr - mu) / sd
        return np.flatnonzero(z > zthr).tolist(), st


def _agg_np(arr: np.ndarray) -> Agg:
    if arr.size == 0:
        return Agg()
    a = arr.astype(np.float64)
    return Agg(arr.size, float(a.sum()), float(a.min()), float(a.max()),
               float((a ** 2).sum()))


def _agg_runs(values: np.ndarray, counts: np.ndarray) -> Agg:
    """Homomorphic aggregate over RLE runs (never materialises the values)."""
    if values.size == 0:
        return Agg()
    c = counts.astype(np.float64)
    return Agg(int(c.sum()), float((values * c).sum()), float(values.min()),
               float(values.max()), float((values * values * c).sum()))


def _windows_np(arr: np.ndarray, window: int) -> List[Agg]:
    out: List[Agg] = []
    if arr.size == 0 or window <= 0:
        return out
    for s in range(0, arr.size, window):
        w = arr[s:s + window]
        if w.size == 0:
            continue
        out.append(Agg(w.size, float(w.sum()), float(w.min()), float(w.max()),
                       float((w ** 2).sum())))
    return out


# --------------------------------------------------------------------------------------
# Prism: segment store + SegmentSketch + optional ShapeIndex
# --------------------------------------------------------------------------------------


@dataclass
class PrismSegment:
    idx: int
    lo: int
    hi: int
    payload: bytes
    sketch_a: Optional[SidecarA]
    sketch_b: Optional[SidecarB] = None
    shape_enabled: bool = True
    codec: str = "chimp128"
    # Mean and population standard deviation of the segment, DERIVED from the
    # sketch's exact sum / exact sum of squares / count when the store is built.
    # Nothing extra is stored on disk: these are functions of sidecar A, so the
    # storage accounting is unchanged; deriving them once per store (instead of
    # once per segment per query) is a pure implementation choice.
    mean_raw: float = 0.0
    std_raw: float = 0.0

    def decode(self) -> np.ndarray:
        n = self.hi - self.lo
        if self.codec == "gorilla":
            return fastcodecs.gorilla_decode(self.payload, n)
        return fastcodecs.chimp128_decode(self.payload, n)


class PrismStore(BaseStore):
    """Prism: per-segment base codec + SegmentSketch + ShapeIndex + segment directory.

    The sketch answers range aggregation without decoding the interior segments; only
    the two boundary segments of a range have to be decoded.  The ShapeIndex provides a
    PROVABLE pruning bound for shape queries (triangle inequality on the PAA
    projection), so a segment that provably cannot match is skipped without decoding.
    """

    name = "Prism"

    def __init__(self, cols: np.ndarray, name: str = "Prism", use_sketch: bool = True,
                 use_shape: bool = False, S: int = S_DEFAULT,
                 codec: str = "chimp128",
                 shape_gate: bool = False, shape_min_rate: float = 0.0,
                 shape_eps: float = 0.05) -> None:
        super().__init__(cols.shape[0])
        self.cols = cols
        self.name = name
        self.use_sketch = use_sketch
        self.use_shape = use_shape
        self.S = S
        self.shape_gate = shape_gate
        self.shape_min_rate = shape_min_rate
        self.shape_eps = shape_eps
        self.shape_prune_rates: Dict[int, List[float]] = {}
        self.codec = codec
        self.segments: Dict[int, List[PrismSegment]] = {}
        self.gmin_gmax: Dict[int, Tuple[float, float]] = {}
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}
        enc = (fastcodecs.gorilla_encode if codec == "gorilla"
               else fastcodecs.chimp128_encode)
        base_bits = 0
        side_bits = 0
        dir_bits = 0
        for c in range(cols.shape[1]):
            arr = cols[:, c].astype(np.float64)
            gmin, gmax = float(arr.min()), float(arr.max())
            self.gmin_gmax[c] = (gmin, gmax)
            segs: List[PrismSegment] = []
            for i, s in enumerate(range(0, self.n, S)):
                e = min(s + S, self.n)
                seg = arr[s:e]
                if codec == "hetero":
                    # per-segment codec choice: the smaller of the two streams
                    p_c = fastcodecs.chimp128_encode(seg)
                    p_g = fastcodecs.gorilla_encode(seg)
                    if len(p_g) < len(p_c):
                        payload, seg_codec = p_g, "gorilla"
                    else:
                        payload, seg_codec = p_c, "chimp128"
                else:
                    payload, seg_codec = enc(seg), codec
                sk = SidecarA.build(seg) if use_sketch else None
                sb = None
                if use_shape:
                    sb = SidecarB.build(seg, k=SHAPE_K, gmin=gmin, gmax=gmax)
                # Derived from the sketch (exact sum / exact sum of squares /
                # count), so the sidecar's storage cost is unchanged.
                m_raw = 0.0
                s_raw = 0.0
                if sk is not None:
                    m_raw = sk.mean_value
                    var = sk.variance_value
                    s_raw = math.sqrt(var) if var > 0.0 else 0.0
                segs.append(PrismSegment(idx=i, lo=s, hi=e, payload=payload,
                                         sketch_a=sk, sketch_b=sb,
                                         shape_enabled=use_shape, codec=seg_codec,
                                         mean_raw=m_raw, std_raw=s_raw))
                base_bits += len(payload) * 8
                if sk is not None:
                    side_bits += sk.bits()
                if sb is not None:
                    side_bits += sb.bits()
                dir_bits += 64 + 64 + 32 + 64
            self.segments[c] = segs
            # -- workload-priced installation of the ShapeIndex --------------------------
            # The shape index only pays off on a segment whose shape the provable
            # bound can actually separate from the workload's match patterns.  Where
            # the bound never fires the index is pure overhead: it costs bits and a
            # per-segment check and saves no decoding.  We therefore measure the
            # pruning rate of every sidecar against a probe population and drop the
            # index where it is ineffective.
            if use_shape and shape_gate:
                rates = self._measure_prune_rates(c, segs)
                self.shape_prune_rates[c] = rates
                for seg, rate in zip(segs, rates):
                    if rate < shape_min_rate:
                        seg.shape_enabled = False
                        if seg.sketch_b is not None:
                            side_bits -= seg.sketch_b.bits()
                            seg.sketch_b = None
        self.base_bits = base_bits
        self.side_bits = side_bits
        self.dir_bits = dir_bits
        self.stored_bytes = (base_bits + side_bits + dir_bits) // 8

    def _measure_prune_rates(self, col: int,
                              segs: List[PrismSegment]) -> List[float]:
        """Pruning rate of every segment's shape index against the probe set.

        The probe population is the workload's match-pattern population: the
        segment-shaped excerpts of the series (a shape query looks for an
        excerpt of the data).  The probe set is supplied by the caller when it
        knows the workload; otherwise the segments themselves are used, which is
        the same population.
        """
        arr = self.cols[:, col].astype(np.float64)
        gmin, gmax = self.gmin_gmax[col]
        rng = gmax - gmin
        if rng <= 0:
            return [0.0] * len(segs)
        full = [s for s in segs if (s.hi - s.lo) == self.S]
        if not full:
            return [0.0] * len(segs)
        block = int(math.ceil(self.S / SHAPE_K))
        radius = self.shape_eps * math.sqrt(self.S)
        # the probe population: one pattern per full-length segment
        probes: List[Tuple[np.ndarray, float]] = []
        for s in full:
            seg = arr[s.lo:s.hi]
            norm = (seg - gmin) / rng
            blk = int(math.ceil(norm.size / SHAPE_K))
            means = np.zeros(SHAPE_K)
            for i in range(SHAPE_K):
                cchunk = norm[i * blk:(i + 1) * blk]
                means[i] = cchunk.mean() if cchunk.size else 0.0
            q = np.clip(np.rint(means * 255.0), 0, 255).astype(np.uint8)
            deq = q.astype(np.float64) / 255.0
            approx = np.repeat(deq, blk)[:norm.size]
            probes.append((deq, float(np.sum((norm - approx) ** 2))))
        sbs = [s.sketch_b for s in segs]
        rates = core_shape_prune_rates(sbs, probes, block, radius)
        return rates

    # -- helpers -----------------------------------------------------------------------
    def _segs(self, col: int) -> List[PrismSegment]:
        return self.segments[col]

    def _decode(self, col: int, seg: PrismSegment, st: QStats) -> np.ndarray:
        """Decode a segment, charging the decode volume only on a cache MISS.

        The cache models a real engine: a segment touched by several windows of the
        same query is materialised once.  Charging on every touch would over-count
        the decode volume of the windowed queries.
        """
        key = (col, seg.idx)
        v = self._cache.get(key)
        if v is None:
            v = seg.decode()
            self._cache[key] = v
            st.decoded_points += v.size
            st.bytes_read += len(seg.payload) + self._sidecar_bytes(seg, False)
            st.items_scanned += 1
        return v

    SIDECAR_DIR_BITS = 224      # (first_ts, last_ts, value_count, bit_offset)

    def _sidecar_bytes(self, seg: PrismSegment, shape: bool) -> int:
        """Bytes read from the storage layer to look at one segment's metadata."""
        b = 0
        if seg.sketch_a is not None:
            b += seg.sketch_a.bits() // 8
        if shape and seg.sketch_b is not None:
            b += seg.sketch_b.bits() // 8
        return b + self.SIDECAR_DIR_BITS // 8

    def _reset_cache(self) -> None:
        self._cache.clear()

    @staticmethod
    def _merge_exact(items: List[Tuple[int, int]]) -> Tuple[int, int]:
        m, e = 0, 0
        for mm, ee in items:
            m, e = _align(m, e, mm, ee)
        return m, e

    def _sketch_agg(self, segs: List[PrismSegment], lo: int, hi: int,
                    st: QStats) -> Tuple[Agg, bool]:
        """Aggregate [lo, hi) from the sketches of fully covered segments.

        Returns (agg, exact) where exact=False means boundary segments must be decoded
        (the caller adds them).
        """
        items: List[Tuple[int, int]] = []
        sqitems: List[Tuple[int, int]] = []
        count = 0
        vmin = math.inf
        vmax = -math.inf
        partial: List[PrismSegment] = []
        for seg in segs:
            if seg.hi <= lo or seg.lo >= hi:
                continue
            st.segments_touched += 1
            if self.use_sketch and seg.lo >= lo and seg.hi <= hi:
                sk = seg.sketch_a
                items.append((sk.sum_mant, sk.sum_exp))
                sqitems.append((sk.sq_mant, sk.sq_exp))
                count += sk.count
                vmin = min(vmin, sk.vmin)
                vmax = max(vmax, sk.vmax)
                st.sketch_hits += 1
            else:
                partial.append(seg)
        if not items and not partial:
            return Agg(), True
        tm, te = self._merge_exact(items)
        qm, qe = self._merge_exact(sqitems)
        total = int_to_float(tm, te)
        if count:
            var = _exact_var(qm, qe, tm, te, count)
        else:
            var = float("nan")
        return Agg(count, total, vmin, vmax, float("nan")), False

    # -- operations --------------------------------------------------------------------
    def agg_range(self, col: int, lo: int, hi: int) -> Tuple[Agg, QStats]:
        self._reset_cache()
        st = QStats()
        segs = self._segs(col)
        sum_items: List[Tuple[int, int]] = []
        sqitems: List[Tuple[int, int]] = []
        count = 0
        vmin = math.inf
        vmax = -math.inf
        for seg in segs:
            if seg.hi <= lo or seg.lo >= hi:
                continue
            st.segments_touched += 1
            st.items_scanned += 1
            if self.use_sketch and seg.lo >= lo and seg.hi <= hi:
                st.bytes_read += self._sidecar_bytes(seg, False)
                sk = seg.sketch_a
                sum_items.append((sk.sum_mant, sk.sum_exp))
                sqitems.append((sk.sq_mant, sk.sq_exp))
                count += sk.count
                vmin = min(vmin, sk.vmin)
                vmax = max(vmax, sk.vmax)
                st.sketch_hits += 1
            else:
                vals = self._decode(col, seg, st)
                sel = vals[max(0, lo - seg.lo):min(vals.size, hi - seg.lo)]
                if sel.size:
                    m, e, qm, qe = exact_accumulate(sel)
                    sum_items.append((m, e))
                    sqitems.append((qm, qe))
                    count += int(sel.size)
                    vmin = min(vmin, float(sel.min()))
                    vmax = max(vmax, float(sel.max()))
        if count == 0:
            return Agg(), st
        tm, te = self._merge_exact(sum_items)
        total = int_to_float(tm, te)
        qm, qe = self._merge_exact(sqitems)
        var = _exact_var(qm, qe, tm, te, count)
        # `sumsq` keeps the exactly-rounded float of the exact sum of squares;
        # `var_override` carries the exactly-derived variance (no float cancellation).
        return Agg(count, total, vmin, vmax, int_to_float(qm, qe), var), st

    def filter_expr(self, col: int, lo: int, hi: int, thr: float,
                    a: float, b: float) -> Tuple[Agg, QStats]:
        """Value filter + affine expression.

        The SegmentSketch's min/max prunes segments that cannot contain a value above
        the threshold; the surviving segments are decoded and filtered exactly.
        """
        self._reset_cache()
        st = QStats()
        segs = self._segs(col)
        count = 0
        total = 0.0
        vmin = math.inf
        vmax = -math.inf
        sumsq = 0.0
        for seg in segs:
            if seg.hi <= lo or seg.lo >= hi:
                continue
            st.segments_touched += 1
            if self.use_sketch and seg.sketch_a.vmax <= thr:
                st.sketch_hits += 1          # pruned by the sketch's min/max
                continue
            vals = self._decode(col, seg, st)
            sel = vals[max(0, lo - seg.lo):min(vals.size, hi - seg.lo)]
            sel = sel[sel > thr]
            if sel.size == 0:
                continue
            e = a * sel + b
            count += int(sel.size)
            total += float(e.sum())
            vmin = min(vmin, float(e.min()))
            vmax = max(vmax, float(e.max()))
            sumsq += float((e ** 2).sum())
        return Agg(count, total, vmin, vmax, sumsq), st

    def window_agg(self, col: int, lo: int, hi: int, window: int) -> Tuple[List[Agg], QStats]:
        """Tumbling-window aggregation.

        The interior segments of a window come from the SegmentSketch; only the
        segments that a window boundary cuts are decoded.  Decoded segments are cached
        for the duration of the query (as a real engine would).  The segments that
        overlap a window are located by index arithmetic, so the cost is
        O(#segments + #overlaps) and not O(#segments x #windows).
        """
        self._reset_cache()
        st = QStats()
        segs = self._segs(col)
        out: List[Agg] = []
        if window <= 0 or hi <= lo:
            return out, st
        for start in range(lo, hi, window):
            end = min(start + window, hi)
            count = 0
            total_m, total_e = 0, 0
            sq_m, sq_e = 0, 0
            vmin = math.inf
            vmax = -math.inf
            i1 = (end - 1) // self.S
            i0 = start // self.S
            for i in range(i0, i1 + 1):
                seg = segs[i]
                st.segments_touched += 1
                st.items_scanned += 1
                if self.use_sketch and seg.lo >= start and seg.hi <= end:
                    st.bytes_read += self._sidecar_bytes(seg, False)
                    sk = seg.sketch_a
                    total_m, total_e = _align(total_m, total_e, sk.sum_mant, sk.sum_exp)
                    sq_m, sq_e = _align(sq_m, sq_e, sk.sq_mant, sk.sq_exp)
                    count += sk.count
                    vmin = min(vmin, sk.vmin)
                    vmax = max(vmax, sk.vmax)
                    st.sketch_hits += 1
                else:
                    vals = self._decode(col, seg, st)
                    sel = vals[max(0, start - seg.lo):min(vals.size, end - seg.lo)]
                    if sel.size:
                        m, e, qm, qe = exact_accumulate(sel)
                        total_m, total_e = _align(total_m, total_e, m, e)
                        sq_m, sq_e = _align(sq_m, sq_e, qm, qe)
                        count += int(sel.size)
                        vmin = min(vmin, float(sel.min()))
                        vmax = max(vmax, float(sel.max()))
            if count == 0:
                continue
            total = int_to_float(total_m, total_e)
            var = _exact_var(sq_m, sq_e, total_m, total_e, count)
            out.append(Agg(count, total, vmin, vmax, int_to_float(sq_m, sq_e), var))
        return out, st

    def filter_window_agg(self, col: int, lo: int, hi: int, window: int, thr: float,
                          a: float, b: float) -> Tuple[List[Agg], QStats]:
        """Value filter + tumbling-window aggregate.

        The segment sketch decides the fate of a segment without decoding it:
          * max <= thr  -> no value passes the filter, the segment contributes
                           nothing at all;
          * min >  thr  -> every value passes, so the segment's exact sum and sum
                           of squares can be reused from the sketch (only for a
                           plain aggregate, i.e. a == 1 and b == 0);
          * otherwise   -> the segment holds values on both sides of the
                           threshold and has to be decoded and filtered.
        A segment that is only partly inside a window is always decoded, because
        the sketch describes the whole segment.
        """
        self._reset_cache()
        st = QStats()
        segs = self._segs(col)
        out: List[Agg] = []
        if window <= 0 or hi <= lo:
            return out, st
        plain = (a == 1.0 and b == 0.0)
        for start in range(lo, hi, window):
            end = min(start + window, hi)
            i0 = start // self.S
            i1 = (end - 1) // self.S
            items: List[Tuple[int, int]] = []
            sqitems: List[Tuple[int, int]] = []
            count = 0
            vmin = math.inf
            vmax = -math.inf
            for i in range(i0, i1 + 1):
                seg = segs[i]
                st.segments_touched += 1
                st.items_scanned += 1
                full = seg.lo >= start and seg.hi <= end
                sk = seg.sketch_a
                if self.use_sketch and full and sk.vmax <= thr:
                    st.sketch_hits += 1
                    st.bytes_read += self._sidecar_bytes(seg, False)
                    continue
                if self.use_sketch and full and plain and sk.vmin > thr:
                    st.sketch_hits += 1
                    st.bytes_read += self._sidecar_bytes(seg, False)
                    items.append((sk.sum_mant, sk.sum_exp))
                    sqitems.append((sk.sq_mant, sk.sq_exp))
                    count += sk.count
                    vmin = min(vmin, sk.vmin)
                    vmax = max(vmax, sk.vmax)
                    continue
                vals = self._decode(col, seg, st)
                sel = vals[max(0, start - seg.lo):min(vals.size, end - seg.lo)]
                sel = sel[sel > thr]
                if sel.size == 0:
                    continue
                e = a * sel + b
                m, ex, qm, qe = exact_accumulate(e)
                items.append((m, ex))
                sqitems.append((qm, qe))
                count += int(sel.size)
                vmin = min(vmin, float(e.min()))
                vmax = max(vmax, float(e.max()))
            if count == 0:
                continue
            tm, te = self._merge_exact(items)
            total = int_to_float(tm, te)
            qm, qe = self._merge_exact(sqitems)
            var = _exact_var(qm, qe, tm, te, count)
            out.append(Agg(count, total, vmin, vmax, int_to_float(qm, qe), var))
        return out, st

    def point(self, col: int, idx: int) -> Tuple[float, QStats]:
        """Point lookup via the segment directory.

        Only the containing segment is touched, and only its prefix up to the
        requested position is decoded: the Chimp128 and Gorilla streams are
        sequential, so the prefix is enough.
        """
        self._reset_cache()
        st = QStats()
        segs = self._segs(col)
        si = idx // self.S
        seg = segs[si]
        off = idx - seg.lo
        st.segments_touched = 1
        st.items_scanned = 1
        st.bytes_read += len(seg.payload) + self._sidecar_bytes(seg, False)
        if seg.codec == "gorilla":
            vals = fastcodecs.gorilla_decode(seg.payload, off + 1)
        else:
            vals = fastcodecs.chimp128_decode(seg.payload, off + 1)
        st.decoded_points += int(vals.size)
        return float(vals[off]), st

    def match(self, col: int, pattern: np.ndarray, eps: float) -> Tuple[List[int], QStats]:
        """Shape query with THREE provable pruning bounds.

        (1) JOINT (mean, std) bound -- ZERO extra bits, needs only the SegmentSketch.
            For length-L vectors x, y with a = x - mx*1 and b = y - my*1 (both zero
            mean), the reverse triangle inequality gives
                ||x - y||^2 = L (mx-my)^2 + ||a-b||^2
                          >= L (mx-my)^2 + (||a|| - ||b||)^2
                          =  L [(mx-my)^2 + (sd_x - sd_y)^2],
            so  ||x - y||^2 / L >= (mx-my)^2 + (sd_x - sd_y)^2.  A segment whose
            lower bound exceeds eps^2 provably contains no match.  mx and sd_x are
            recovered from the sketch's EXACT sum and EXACT sum of squares.

        (2) RANGE bound -- ZERO extra bits.  If the pattern's value range and the
            segment's value range are separated by more than eps in the query's
            normalisation, then every element pair differs by more than eps, so
            ||x - y||_2 > eps*sqrt(L) and the segment cannot match.

        (3) PAA bound -- the ShapeIndex's triangle inequality:
            || s - q ||_2  >=  sqrt(bs) * || deq(PAA(s)) - deq(PAA(q)) ||_2
                              - sqrt(e_s) - sqrt(e_q)
        where bs is the number of values per PAA block and e_x = || x - deq(PAA(x)) ||^2
        is the residual energy that the sidecar stores.

        All three skip the segment WITHOUT decoding it; only the surviving
        segments are decoded and compared exactly.  The bounds are compared in
        raw units against eps*rng (the query distance is measured on the
        normalised arrays, and normalising divides every difference by rng);
        a relative safety margin of 1e-9 absorbs the rounding of the
        normalisation and of the float64 arithmetic.
        """
        self._reset_cache()
        st = QStats()
        segs = self._segs(col)
        gmin, gmax = self.gmin_gmax[col]
        rng = gmax - gmin
        L = pattern.size
        hits: List[int] = []
        if rng <= 0 or L == 0:
            return hits, st
        qn = (pattern - gmin) / rng
        q_paa, q_energy = _paa(qn, SHAPE_K)
        block = int(math.ceil(L / SHAPE_K))
        radius = eps * math.sqrt(L)
        # The pattern's own statistics, once per query.
        q_mean = float(pattern.mean())
        q_std = float(math.sqrt(max(float(np.var(pattern)), 0.0)))
        q_min = float(pattern.min())
        q_max = float(pattern.max())
        raw_radius = eps * rng
        raw_radius_sq = raw_radius * raw_radius
        margin = 1.0 + 1e-9
        for seg in segs:
            if (seg.hi - seg.lo) != L:
                continue          # only full-length segments align with the query window
            st.segments_touched += 1
            st.items_scanned += 1
            st.bytes_read += self._sidecar_bytes(seg, self.use_shape)
            if self.use_sketch and seg.sketch_a is not None:
                # (1) JOINT (mean, std) bound -- free with the SegmentSketch.
                dm = seg.mean_raw - q_mean
                ds = seg.std_raw - q_std
                if (dm * dm + ds * ds) > raw_radius_sq * margin:
                    st.sketch_hits += 1     # provably no match: skipped without decoding
                    continue
                # (2) RANGE bound -- also free (the min/max are in the sketch).
                gap = max(0.0, max(q_min - seg.sketch_a.vmax,
                                   seg.sketch_a.vmin - q_max))
                if gap > raw_radius * margin:
                    st.sketch_hits += 1
                    continue
            if self.use_shape and seg.sketch_b is not None:
                sb_paa = seg.sketch_b.dequant()
                k = min(len(sb_paa), len(q_paa))
                dist = math.sqrt(block) * float(np.linalg.norm(sb_paa[:k] - q_paa[:k]))
                bound = (dist - math.sqrt(max(seg.sketch_b.energy, 0.0))
                         - math.sqrt(max(q_energy, 0.0)))
                if bound > radius:
                    st.sketch_hits += 1     # provably no match: skipped without decoding
                    continue
            vals = self._decode(col, seg, st)
            sn = (vals - gmin) / rng
            d = float(np.linalg.norm(sn - qn) / math.sqrt(L))
            if d <= eps:
                hits.append(seg.lo)
        return hits, st

    def anom(self, col: int, zthr: float) -> Tuple[List[int], QStats]:
        """Anomaly scan with segment-level pruning.

        The global mean/variance come from the SegmentSketch (exact sum / sumsq over
        all segments, no decoding).  A segment whose min/max cannot reach the z
        threshold is skipped WITHOUT decoding.
        """
        self._reset_cache()
        st = QStats()
        segs = self._segs(col)
        # global statistics from the sketches
        sm = se = 0
        qm = qe = 0
        count = 0
        for seg in segs:
            st.segments_touched += 1
            st.items_scanned += 1
            st.bytes_read += self._sidecar_bytes(seg, False)
            if self.use_sketch:
                sm, se = _align(sm, se, seg.sketch_a.sum_mant, seg.sketch_a.sum_exp)
                qm, qe = _align(qm, qe, seg.sketch_a.sq_mant, seg.sketch_a.sq_exp)
                count += seg.sketch_a.count
            else:
                vals = self._decode(col, seg, st)
                m, e, qm2, qe2 = exact_accumulate(vals)
                sm, se = _align(sm, se, m, e)
                qm, qe = _align(qm, qe, qm2, qe2)
                count += vals.size
        if count == 0:
            return [], st
        mean = int_to_float(sm, se) / count
        var = _exact_var(qm, qe, sm, se, count)
        sd = math.sqrt(max(var, 0.0))
        hits: List[int] = []
        if sd <= 0:
            return hits, st
        thr = zthr * sd
        for seg in segs:
            st.segments_touched += 1
            if self.use_sketch:
                sk = seg.sketch_a
                # the segment's z-extreme is bounded by its min/max
                if max(abs(sk.vmax - mean), abs(sk.vmin - mean)) <= thr:
                    st.sketch_hits += 1     # pruned without decoding
                    continue
            vals = self._decode(col, seg, st)
            z = np.abs(vals - mean) / sd
            for off in np.flatnonzero(z > zthr).tolist():
                hits.append(seg.lo + off)
        return hits, st


def Fraction_of(m: int, e: int):
    """Exact value m * 2**e as a Fraction."""
    from fractions import Fraction
    if e < 0:
        return Fraction(m, 1 << -e)
    return Fraction(m * (1 << e))

def _exact_var(qm: int, qe: int, tm: int, te: int, count: int) -> float:
    """Correctly-rounded float64 of the exact population variance
        (sumsq - sum**2 / count) / count,   sumsq = qm*2**qe,  sum = tm*2**te.

    Same value as the Fraction expression it replaces
    (`_frac_to_float((Fraction_of(qm,qe) - Fraction_of(tm,te)**2/count)/count)`):
    the rational (count*sumsq - sum**2) / count**2 is formed on a common binary
    exponent with plain Python ints and rounded once by int/int true division,
    which CPython rounds correctly -- exactly what float(Fraction) does, minus the
    gcd normalisation of every intermediate Fraction.
    """
    if count <= 0:
        return float("nan")
    E = min(qe, 2 * te)
    num = ((count * qm) << (qe - E)) - ((tm * tm) << (2 * te - E))
    if num == 0:
        return 0.0
    den = count * count
    try:
        if E >= 0:
            return (num << E) / den
        return num / (den << -E)
    except OverflowError:
        return math.inf if num > 0 else -math.inf

def _paa(x: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    """Quantised block-average projection + exact residual energy."""
    n = x.size
    block = int(math.ceil(n / k))
    means = np.zeros(k)
    for i in range(k):
        c = x[i * block:(i + 1) * block]
        means[i] = c.mean() if c.size else 0.0
    q = np.clip(np.rint(means * 255.0), 0, 255).astype(np.uint8)
    deq = q.astype(np.float64) / 255.0
    approx = np.repeat(deq, block)[:n]
    energy = float(np.sum((x - approx) ** 2))
    return deq, energy


# --------------------------------------------------------------------------------------
# store factory
# --------------------------------------------------------------------------------------


def build_stores(cols: np.ndarray, S: int = S_DEFAULT,
                 use_shape: bool = True) -> Dict[str, BaseStore]:
    """All storage formats of the evaluation, built over the same columns."""
    out: Dict[str, BaseStore] = {}
    out["Uncompressed"] = UncompressedStore(cols)
    out["IoTDB"] = IoTDBStore(cols)
    out["CompressIoTDB-NoLate"] = CompressIoTDBStore(cols, late=False)
    out["CompressIoTDB"] = CompressIoTDBStore(cols, late=True)
    out["Prism"] = PrismStore(cols, "Prism", use_sketch=True, use_shape=use_shape, S=S)
    out["Prism-NoSketch"] = PrismStore(cols, "Prism-NoSketch", use_sketch=False,
                                       use_shape=False, S=S)
    out["Prism-SketchOnly"] = PrismStore(cols, "Prism-SketchOnly", use_sketch=True,
                                         use_shape=False, S=S)
    return out


def execute(store: BaseStore, spec: QuerySpec, ctx: dict) -> Tuple[object, QStats]:
    """Run one query against one store; returns (result, stats)."""
    if spec.kind == "range_agg":
        agg = Agg()
        st = QStats()
        for c in spec.cols:
            a, s = store.agg_range(c, spec.lo, spec.hi)
            agg = agg + a
            st = st + s
        return agg.digest(), st
    if spec.kind == "filter_expr":
        agg = Agg()
        st = QStats()
        for c in spec.cols:
            a, s = store.filter_expr(c, spec.lo, spec.hi, spec.thr,
                                      spec.expr_a, spec.expr_b)
            agg = agg + a
            st = st + s
        return agg.digest(), st
    if spec.kind == "window_agg":
        st = QStats()
        outs: List[List[Agg]] = []
        for c in spec.cols:
            o, s = store.window_agg(c, spec.lo, spec.hi, spec.window)
            outs.append(o)
            st = st + s
        # flat list of per-window triples, one group per column
        flat: List[Tuple[int, float, float]] = []
        for col in outs:
            for a in col:
                flat.append((a.count, round(a.total, 6), round(a.var, 6)))
        return flat, st
    if spec.kind == "filter_window_agg":
        # value filter + tumbling-window aggregate (Q7 operator class)
        st = QStats()
        outs: List[List[Agg]] = []
        for c in spec.cols:
            o, s = store.filter_window_agg(c, spec.lo, spec.hi, spec.window,
                                           spec.thr, spec.expr_a, spec.expr_b)
            outs.append(o)
            st = st + s
        flat = []
        for col in outs:
            for a in col:
                flat.append((a.count, round(a.total, 6), round(a.var, 6)))
        return flat, st
    if spec.kind == "point":
        vals = []
        st = QStats()
        for c in spec.cols:
            v, s = store.point(c, ctx["point_idx"])
            vals.append(v)
            st = st + s
        return vals, st
    if spec.kind == "match":
        st = QStats()
        hits = []
        for c in spec.cols:
            h, s = store.match(c, ctx["pattern"], spec.eps)
            hits.append(h)
            st = st + s
        return hits, st
    if spec.kind == "anom":
        st = QStats()
        hits = []
        for c in spec.cols:
            h, s = store.anom(c, spec.zthr)
            hits.append(h)
            st = st + s
        return hits, st
    raise ValueError(f"unknown query kind {spec.kind}")
