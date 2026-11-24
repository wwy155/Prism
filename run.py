"""Run the thirteen query classes of the paper on one dataset with the default
Prism configuration (segment size 512, Chimp128 segments, SegmentSketch and ShapeIndex).

Every format answers the same queries; each answer is compared with the answer over
the uncompressed data.  Each query is executed `--warmup` times, then `--reps` times,
and the mean latency of the timed executions is reported.

Usage:
    python run.py --csv data/weather.csv
    python run.py --csv data/ETTh1.csv --attrs 6 --out results/etth1.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "prism"))

from engine import (CompressIoTDBStore, IoTDBStore, PrismStore,  # noqa: E402
                    QuerySpec, UncompressedStore, execute)

S = 512
FORMATS = ["Uncompressed", "IoTDB", "CompressIoTDB-NoLate", "CompressIoTDB", "Prism"]


def load_csv(path: str, attrs: int) -> Tuple[np.ndarray, np.ndarray]:
    """One series per attribute column; the timestamp column is dropped."""
    df = pd.read_csv(path, low_memory=False)
    names = [c for c in df.columns if c.lower() not in ("date", "time", "timestamp")]
    v = df[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    v = v[:, :attrs]
    v = v[np.isfinite(v).all(axis=1)]
    return np.arange(len(v), dtype=np.int64), v


def build_queries(ts: np.ndarray, cols: np.ndarray) -> List[Tuple[QuerySpec, dict]]:
    """Q1-Q10: filter/expression, window and whole-range aggregates; Q11 point lookup;
    Q12 shape query; Q13 anomaly scan."""
    n = ts.size
    k = cols.shape[1]
    day = max(1, n // 24)
    short = max(1, n // 8)
    thr = float(np.median(cols[:, 0]))
    q: List[Tuple[QuerySpec, dict]] = []
    q.append((QuerySpec("Q1", "filter_expr", cols=(0,), lo=0, hi=n, thr=thr,
                        expr_a=9.0 / 5.0, expr_b=32.0), {}))
    q.append((QuerySpec("Q2", "window_agg", cols=(0, 1 % k), lo=0, hi=short,
                        window=max(1, short // 16)), {}))
    q.append((QuerySpec("Q3", "range_agg", cols=(0,), lo=0, hi=n), {}))
    q.append((QuerySpec("Q4", "range_agg", cols=tuple(range(min(2, k))), lo=0, hi=n), {}))
    q.append((QuerySpec("Q5", "window_agg", cols=(0,), lo=0, hi=n, window=day), {}))
    q.append((QuerySpec("Q6", "window_agg", cols=(1 % k,), lo=0, hi=n, window=day), {}))
    q.append((QuerySpec("Q7", "filter_window_agg", cols=(0,), lo=0, hi=n, window=day,
                        thr=thr, expr_a=1.0, expr_b=0.0), {}))
    q.append((QuerySpec("Q8", "window_agg", cols=(min(1, k - 1),), lo=0, hi=n,
                        window=day), {}))
    q.append((QuerySpec("Q9", "window_agg", cols=(0,), lo=0, hi=n, window=day), {}))
    q.append((QuerySpec("Q10", "window_agg", cols=(min(1, k - 1),), lo=0, hi=n,
                        window=day), {}))
    q.append((QuerySpec("Q11", "point", cols=(0,)), {"point_idx": n // 2}))
    seg = (n // 3) // S
    pat = cols[seg * S:seg * S + S, 0].copy()
    q.append((QuerySpec("Q12", "match", cols=(0,), eps=0.05), {"pattern": pat}))
    q.append((QuerySpec("Q13", "anom", cols=(0,), zthr=3.0), {}))
    return q


def measure(store, spec, ctx, reps: int, warmup: int):
    for _ in range(warmup):
        execute(store, spec, ctx)
    total = 0.0
    res = st = None
    for _ in range(reps):
        t0 = time.perf_counter()
        res, st = execute(store, spec, ctx)
        total += time.perf_counter() - t0
    return 1000.0 * total / reps, st, res


def same_answer(a, b, rel: float = 1e-9) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_answer(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return a == b or abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)
    return a == b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="dataset file, one column per attribute")
    ap.add_argument("--attrs", type=int, default=6, help="number of attribute columns")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default=None, help="output CSV (default results/<name>.csv)")
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.csv))[0]
    out = args.out or os.path.join(HERE, "results", name + ".csv")
    ts, cols = load_csv(args.csv, args.attrs)
    print(f"{name}: {cols.shape[0]} timestamps x {cols.shape[1]} series")

    stores = {
        "Uncompressed": UncompressedStore(cols),
        "IoTDB": IoTDBStore(cols),
        "CompressIoTDB-NoLate": CompressIoTDBStore(cols, late=False),
        "CompressIoTDB": CompressIoTDBStore(cols, late=True),
        "Prism": PrismStore(cols, "Prism", use_sketch=True, use_shape=True, S=S),
    }
    print("\nstored bits per value")
    for f in FORMATS:
        print(f"  {f:22s} {8.0 * stores[f].stored_bytes / cols.size:7.2f}")

    rows = []
    print(f"\n{'query':5s} {'format':22s} {'latency(ms)':>12s} {'items':>9s} "
          f"{'bytes':>10s} {'decoded':>9s}  answer")
    for spec, ctx in build_queries(ts, cols):
        ref = None
        for f in FORMATS:
            lat, st, res = measure(stores[f], spec, ctx, args.reps, args.warmup)
            if ref is None:
                ref = res
            ok = same_answer(ref, res)
            rows.append([name, spec.qid, f, f"{lat:.6f}", st.items_scanned, st.bytes_read,
                         st.decoded_points, int(ok)])
            print(f"{spec.qid:5s} {f:22s} {lat:12.3f} {st.items_scanned:9d} "
                  f"{st.bytes_read:10d} {st.decoded_points:9d}  {'ok' if ok else 'DIFF'}")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "query", "format", "latency_ms", "items", "bytes",
                    "decoded", "same_answer"])
        w.writerows(rows)
    print(f"\nwrote {out}")
    return 0 if all(r[-1] == 1 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
