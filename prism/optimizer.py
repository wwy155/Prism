"""Workload-priced sidecar planner.

A segment i can be stored with any candidate c = (base codec, sidecar level).  For a
query class q the latency of a segment under candidate c is modelled as

    lat_q(i, c) = alpha_q + beta_q * d_q(i, c) + delta_q * ops_q(i, c)

where d_q is the number of values the class decodes on the segment and ops_q the number
of sidecar operations.  With the workload distribution w (decayed query-class counts)
the planner maximizes

    sum_i sum_q w_q * (L_q^ref - lat_q(i, c_i))   s.t.   sum_i bits(i, c_i) <= B,

a multiple-choice knapsack.  Its Lagrangian relaxation with space price gamma
separates into independent per-segment choices

    c_i = argmax_c { sum_q w_q * gain_q(i, c) - gamma * bits(i, c) }.

For an explicit space limit B, gamma is bisected and the plan is completed greedily,
which keeps it within the gain of a single segment of the integer optimum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

QUERY_CLASSES = ("agg", "probe", "match", "anom")

BASE_CODECS = ("chimp128", "gorilla", "rledict", "plain")

# sidecar levels: (use_A, use_B, use_C)
SIDECAR_LEVELS = {
    "none": (False, False, True),    # sidecar C (segment directory) is always needed
    "A": (True, False, True),
    "AC": (True, False, True),
    "ABC": (True, True, True),
}
# "none" and "AC" differ: "none" has no aggregate sketch, "AC" has it
SIDECAR_LEVELS = {
    "none": (False, False, True),
    "A": (True, False, True),
    "AB": (True, True, True),
    "ABC": (True, True, True),
}

@dataclass(frozen=True)
class Candidate:
    """A (base codec, sidecar level) pair -- one of the |C| = 16 layout choices."""

    codec: str
    sidecar: str

    @property
    def key(self) -> str:
        return f"{self.codec}+{self.sidecar}"


CANDIDATES: List[Candidate] = [Candidate(c, s) for c in BASE_CODECS for s in SIDECAR_LEVELS]


@dataclass
class SegmentProfile:
    """Everything the optimiser needs to know about one segment."""

    idx: int
    n_values: int
    # bits of the base stream under each codec
    base_bits: Dict[str, float]
    # measured decode time (seconds) of the segment under each codec
    decode_time: Dict[str, float]
    # sidecar bits
    sketch_a_bits: int
    sketch_b_bits: int
    directory_bits: int
    # value range, used by the match/anom pruning models
    vmin: float
    vmax: float
    # true (verified) match: does the segment contain a match of the query pattern?
    contains_match: bool = False
    # z-score range of the segment under the global statistics
    zmax: float = 0.0
    # is the segment fully covered by the query window?
    fully_covered: bool = False
    # does the segment contain the probed point?
    contains_point: bool = False


@dataclass
class CostModel:
    """Fitted latency model: lat_q = alpha_q + beta_q * decoded_values + sidecar_cost_q."""

    alpha: Dict[str, float] = field(default_factory=dict)
    beta: Dict[str, float] = field(default_factory=dict)
    sketch_cost: Dict[str, float] = field(default_factory=dict)
    fit_error: Dict[str, float] = field(default_factory=dict)
    n_points: Dict[str, int] = field(default_factory=dict)

    def predict(self, q: str, decoded: float, sidecar_ops: float) -> float:
        return self.alpha.get(q, 0.0) + self.beta.get(q, 0.0) * decoded + \
            self.sketch_cost.get(q, 0.0) * sidecar_ops

    def to_dict(self) -> dict:
        return {
            "alpha": dict(self.alpha),
            "beta": dict(self.beta),
            "sketch_cost": dict(self.sketch_cost),
            "fit_error_rel": dict(self.fit_error),
            "n_points": dict(self.n_points),
        }


def fit_cost_model(samples: Sequence[Tuple[str, float, float, float]]) -> CostModel:
    """Least-squares fit of lat_q = alpha_q + beta_q * decoded + delta_q * sidecar_ops.

    `samples` are (query_class, decoded_values, sidecar_ops, measured_seconds).
    """
    cm = CostModel()
    for q in QUERY_CLASSES:
        xs = [(d, s, t) for (qq, d, s, t) in samples if qq == q]
        if len(xs) < 3:
            cm.alpha[q] = 0.0
            cm.beta[q] = 0.0
            cm.sketch_cost[q] = 0.0
            cm.fit_error[q] = float("nan")
            cm.n_points[q] = len(xs)
            continue
        A = np.array([[1.0, d, s] for (d, s, _) in xs], dtype=np.float64)
        y = np.array([t for (_, _, t) in xs], dtype=np.float64)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        denom = np.abs(y).mean() if np.abs(y).mean() > 0 else 1.0
        cm.alpha[q] = float(coef[0])
        cm.beta[q] = float(coef[1])
        cm.sketch_cost[q] = float(coef[2])
        cm.fit_error[q] = float(np.abs(pred - y).mean() / denom)
        cm.n_points[q] = len(xs)
    return cm


def candidate_space_bits(seg: SegmentProfile, cand: Candidate) -> float:
    use_a, use_b, _ = SIDECAR_LEVELS[cand.sidecar]
    bits = seg.base_bits[cand.codec]
    if use_a:
        bits += seg.sketch_a_bits
    if use_b:
        bits += seg.sketch_b_bits
    bits += seg.directory_bits
    return bits


def candidate_sidecar_ops(seg: SegmentProfile, cand: Candidate) -> float:
    """How many sidecar operations the candidate performs for the 4 query classes."""
    use_a, use_b, _ = SIDECAR_LEVELS[cand.sidecar]
    # agg: one sketch read per fully covered segment
    agg = 1.0 if (use_a and seg.fully_covered) else 0.0
    # probe: no sidecar help
    probe = 0.0
    # match: one prune test per segment
    match = 1.0 if use_b else 0.0
    # anom: one prune test per segment (uses min/max from A)
    anom = 1.0 if use_a else 0.0
    return agg + probe + match + anom


def candidate_decoded_values(seg: SegmentProfile, cand: Candidate) -> Dict[str, float]:
    """How many values each query class must decode under this candidate."""
    use_a, use_b, _ = SIDECAR_LEVELS[cand.sidecar]
    out = {}
    # range aggregation over a window that fully covers the segment
    out["agg"] = 0.0 if use_a else float(seg.n_values)
    # point query: always needs the segment's values
    out["probe"] = float(seg.n_values) if seg.contains_point else 0.0
    # pattern matching: with sidecar B the segment can be pruned; without it, decoded
    if use_b:
        out["match"] = 0.0 if not seg.contains_match else float(seg.n_values)
    else:
        out["match"] = float(seg.n_values)
    # anomaly scan: with sidecar A the segment can be pruned by its min/max
    if use_a:
        out["anom"] = 0.0 if seg.zmax <= 1.0 else float(seg.n_values)
    else:
        out["anom"] = float(seg.n_values)
    return out


def gain_of(seg: SegmentProfile, cand: Candidate, cm: CostModel, w: Dict[str, float],
            ref_lat: Dict[str, float]) -> float:
    """Weighted latency saving of candidate `cand` over the reference configuration."""
    dv = candidate_decoded_values(seg, cand)
    so = candidate_sidecar_ops(seg, cand)
    total = 0.0
    for q in QUERY_CLASSES:
        lat = cm.predict(q, dv[q], so if q in ("agg", "match", "anom") else 0.0)
        total += w.get(q, 0.0) * (ref_lat.get(q, 0.0) - lat)
    return total


def lagrangian_select(segs: Sequence[SegmentProfile], cm: CostModel, w: Dict[str, float],
                      ref_lat: Dict[str, float], gamma: float,
                      budget_bits: Optional[float] = None) -> List[Candidate]:
    """Per-segment Lagrangian choice (multi-choice knapsack relaxation).

    If `budget_bits` is given, gamma is first bisected so that the resulting
    configuration respects the budget (the space price is what enforces the budget).
    """
    def solve(g: float) -> List[Candidate]:
        chosen: List[Candidate] = []
        for seg in segs:
            best = None
            best_val = -math.inf
            for codec in BASE_CODECS:
                for side in SIDECAR_LEVELS:
                    cand = Candidate(codec, side)
                    val = gain_of(seg, cand, cm, w, ref_lat) - g * candidate_space_bits(seg, cand)
                    if val > best_val:
                        best_val = val
                        best = cand
            chosen.append(best or Candidate("chimp128", "none"))
        return chosen

    if budget_bits is None:
        return solve(gamma)

    # Budget mode: find the SMALLEST space price whose resulting plan fits the
    # budget.  gamma has the physical unit "seconds per bit": the latency
    # gains are of the order 1e-5 s while a plan differs by ~1e4 bits, so the
    # relevant range is 1e-12 .. 1e-6.  The bracket is searched geometrically from a
    # small price upwards, and the returned price is the largest one that
    # still fits the budget (the cheapest plan that satisfies it is found by
    # the bisection below).
    def space_of(g: float) -> float:
        return sum(candidate_space_bits(s, c) for s, c in zip(segs, solve(g)))

    lo, hi = 1e-12, 1e-6
    if space_of(lo) <= budget_bits:
        return solve(lo)
    # grow hi until the plan fits
    for _ in range(80):
        if space_of(hi) <= budget_bits:
            break
        lo = hi
        hi *= 10.0
        if hi > 1e6:
            break
    # bisect on the price: the plan shrinks as the price grows, so the
    # largest price that still fits the budget is the one we want
    for _ in range(80):
        mid = (lo * hi) ** 0.5
        if space_of(mid) <= budget_bits:
            hi = mid
        else:
            lo = mid
    return _complete(segs, solve(hi), solve(lo), cm, w, ref_lat, budget_bits)


def _complete(segs: Sequence[SegmentProfile], plan_fit: List[Candidate],
              plan_over: List[Candidate], cm: CostModel, w: Dict[str, float],
              ref_lat: Dict[str, float], budget_bits: float) -> List[Candidate]:
    """Greedy completion of the bisected plan (multiple-choice knapsack).

    `plan_fit` is the per-segment minimiser at the smallest price that fits the
    budget and `plan_over` the minimiser just below it.  The segments on which
    the two differ are upgraded in order of weighted latency gain per extra bit
    while the budget allows.  Every segment but at most one is then integral in
    the LP optimum of the relaxation, so the completed plan is within the gain of
    a single segment upgrade of the integer optimum (Sinha & Zoltners, 1979).
    """
    out = list(plan_fit)
    used = sum(candidate_space_bits(s, c) for s, c in zip(segs, out))
    ups = []
    for i, (s, a, b) in enumerate(zip(segs, plan_fit, plan_over)):
        if a == b:
            continue
        db = candidate_space_bits(s, b) - candidate_space_bits(s, a)
        dg = gain_of(s, b, cm, w, ref_lat) - gain_of(s, a, cm, w, ref_lat)
        if dg <= 0:
            continue
        ups.append((dg / db if db > 0 else math.inf, i, db))
    ups.sort(key=lambda t: -t[0])
    for _, i, db in ups:
        if used + db <= budget_bits:
            out[i] = plan_over[i]
            used += db
    return out


# --------------------------------------------------------------------------------------
# online load model: decayed query-class counts
# --------------------------------------------------------------------------------------


class LoadTracker:
    """Decayed query-class counters with half-life tau.

    w_q is the (normalised) probability that the next query belongs to class q.
    The tracker is updated online with the classes of the queries actually observed;
    it is never fitted to the evaluation workload.
    """

    def __init__(self, tau: float = 64.0) -> None:
        self.tau = tau
        self.counts = {q: 0.0 for q in QUERY_CLASSES}
        self.t = 0.0

    def observe(self, q: str) -> None:
        self.t += 1.0
        decay = 0.5 ** (1.0 / self.tau)
        for k in self.counts:
            self.counts[k] *= decay
        self.counts[q] += 1.0

    def w(self) -> Dict[str, float]:
        tot = sum(self.counts.values())
        if tot <= 0:
            return {q: 1.0 / len(QUERY_CLASSES) for q in QUERY_CLASSES}
        return {q: self.counts[q] / tot for q in QUERY_CLASSES}

    def kl_to(self, other: Dict[str, float]) -> float:
        w = self.w()
        s = 0.0
        for q in QUERY_CLASSES:
            if w[q] > 0:
                s += w[q] * math.log(w[q] / max(other.get(q, 1e-12), 1e-12))
        return s
