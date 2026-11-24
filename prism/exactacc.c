/*
 * exactacc.c -- exact per-exponent group sums for Prism's exact aggregation.
 *
 * Every finite float64 is decomposed exactly as v = mant * 2^exp with
 * frexp(): m in [0.5, 1), mant = m * 2^53 (an integer, |mant| < 2^53) and
 * exp = e - 53.  This is the same decomposition as the numpy path in core.py.
 * Values are bucketed by exp; inside a bucket the mantissas and the squared
 * mantissas are summed in signed 128-bit integers, which is exact for up to
 * 2^20 values per call (|mant|^2 < 2^106, 2^106 * 2^20 < 2^127).  The Python
 * caller recombines the (few) buckets in ascending exponent order with exact
 * big-integer shifts, so the result is the exact sum and the exact sum of
 * squares.
 *
 * Build: gcc -O2 -shared -fPIC -o libexactacc.so exactacc.c -lm
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#define EXP_OFFSET 1200
#define N_BUCKETS 2400
#define MAX_N (1L << 20)

static __int128 g_sum[N_BUCKETS];
static __int128 g_sq[N_BUCKETS];
static unsigned char g_used[N_BUCKETS];

/*
 * Returns the number of occupied buckets written to the output arrays (in
 * ascending exponent order), -1 if a value is not finite, -2 if n exceeds the
 * exactness limit (the caller splits the array), -3 if cap is too small.
 */
long exact_groups(const double *x, long n, int64_t *exps,
                  int64_t *s_hi, uint64_t *s_lo,
                  int64_t *q_hi, uint64_t *q_lo, long cap)
{
    if (n > MAX_N)
        return -2;
    int lo = N_BUCKETS, hi = -1;
    for (long i = 0; i < n; i++) {
        double v = x[i];
        if (!isfinite(v)) {
            for (int b = lo; b <= hi; b++) { g_sum[b] = 0; g_sq[b] = 0; g_used[b] = 0; }
            return -1;
        }
        int e;
        double m = frexp(v, &e);
        int64_t mant = (int64_t)(m * 9007199254740992.0);   /* exact: m has 53 bits */
        int b = (e - 53) + EXP_OFFSET;
        __int128 mm = (__int128)mant;
        g_sum[b] += mm;
        g_sq[b] += mm * mm;
        g_used[b] = 1;
        if (b < lo) lo = b;
        if (b > hi) hi = b;
    }
    long g = 0;
    for (int b = lo; b <= hi; b++) {
        if (!g_used[b])
            continue;
        if (g >= cap)
            return -3;
        exps[g] = (int64_t)(b - EXP_OFFSET);
        __int128 s = g_sum[b], q = g_sq[b];
        s_hi[g] = (int64_t)(s >> 64);
        s_lo[g] = (uint64_t)s;
        q_hi[g] = (int64_t)(q >> 64);
        q_lo[g] = (uint64_t)q;
        g_sum[b] = 0;
        g_sq[b] = 0;
        g_used[b] = 0;
        g++;
    }
    return g;
}
