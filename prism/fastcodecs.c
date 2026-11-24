/* fastcodecs.c -- C ports of the Chimp128 and Gorilla codecs.
 *
 * The streams are byte-identical to the reference Java implementation of Chimp:
 *   gr/aueb/delorean/chimp/ChimpN.java                -> Chimp128 (ChimpN(128))
 *   fi/iki/yak/ts/compression/gorilla/Compressor.java -> Gorilla
 *
 * Build: gcc -O3 -shared -fPIC -o libfastcodecs.so fastcodecs.c
 */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define MASK64 0xFFFFFFFFFFFFFFFFULL
static const uint64_t NAN_LONG = 0x7ff8000000000000ULL;

/* gr.aueb.delorean.chimp.ChimpN: leadingRepresentation (64 entries) */
static const int LEADING_REPRESENTATION[64] = {
    0,0,0,0,0,0,0,0, 1,1,1,1,2,2,2,2, 3,3,4,4,5,5,6,6,
    7,7,7,7,7,7,7,7, 7,7,7,7,7,7,7,7, 7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7, 7,7,7,7,7,7,7,7
};
/* gr.aueb.delorean.chimp.ChimpN: leadingRound (64 entries) */
static const int LEADING_ROUND[64] = {
    0,0,0,0,0,0,0,0, 8,8,8,8,12,12,12,12, 16,16,18,18,20,20,22,22,
    24,24,24,24,24,24,24,24, 24,24,24,24,24,24,24,24, 24,24,24,24,24,24,24,24,
    24,24,24,24,24,24,24,24, 24,24,24,24,24,24,24,24
};
/* ChimpNDecompressor.leadingRepresentation (8 entries) */
static const int LEADING_FROM_CODE[8] = {0, 8, 12, 16, 18, 20, 22, 24};

static inline int nlz64(uint64_t x) { return x ? __builtin_clzll(x) : 64; }
static inline int ntz64(uint64_t x) { return x ? __builtin_ctzll(x) : 64; }

/* ------------------------------------------------------------------ bit writer */
typedef struct { uint8_t* buf; size_t nbytes; uint64_t acc; int nacc; } bw_t;

static inline void bw_init(bw_t* w, uint8_t* buf, size_t cap) {
    w->buf = buf; w->nbytes = 0; w->acc = 0; w->nacc = 0; (void)cap;
}
static inline void bw_put(bw_t* w, uint64_t v, int nbits) {
    if (nbits <= 0) return;
    if (nbits > 32) {                 /* keep acc from overflowing */
        int hi = nbits - 32;
        bw_put(w, v >> 32, hi);
        bw_put(w, (uint32_t)v, 32);
        return;
    }
    v &= (1ULL << nbits) - 1ULL;
    w->acc = (w->acc << nbits) | v;
    w->nacc += nbits;
    while (w->nacc >= 8) {
        w->nacc -= 8;
        w->buf[w->nbytes++] = (uint8_t)((w->acc >> w->nacc) & 0xFF);
    }
}
static inline size_t bw_flush(bw_t* w) {
    if (w->nacc > 0) {
        w->buf[w->nbytes++] = (uint8_t)((w->acc << (8 - w->nacc)) & 0xFF);
        w->nacc = 0;
    }
    return w->nbytes;
}

/* ------------------------------------------------------------------ bit reader */
/* An eight-byte window is kept in a register.  The stream is MSB first inside a
   byte and the bytes follow each other, so with W holding bytes[byte..byte+7]
   big endian, the bit at stream position bitpos is W's bit (63 - off) where
   off = bitpos & 7.  A read of `nbits` bits is therefore the field
   W[64-off-nbits .. 63-off], right aligned.

   The previous reader walked the buffer one byte at a time, which cost ~114
   ns/value on Chimp128 and would have made the latency table measure this port
   instead of the storage format.  The bit order is unchanged, so the decoded
   output stays byte-identical to the previous reader and to the official Java
   implementation; scripts/prism/bitreader_diff.c proves the equivalence on
   random bit streams and scripts/prism/validate_fastcodecs.py checks it against
   the official Java implementation. */
typedef struct {
    const uint8_t* buf;
    size_t nbytes;
    size_t bitpos;      /* absolute bit position of the next bit */
    uint64_t win;       /* bytes[winbyte .. winbyte+7], big endian */
    uint8_t wnext;      /* bytes[winbyte+8], the byte right after the window */
    size_t winbyte;
} br_t;

/* eight bytes at `byte`, big endian, zero padded past the end of the stream */
static inline uint64_t br_load64(const uint8_t* buf, size_t nbytes, size_t byte) {
    uint64_t v;
    size_t avail = (byte < nbytes) ? (nbytes - byte) : 0;
    if (avail >= 8) {
        memcpy(&v, buf + byte, 8);
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
        v = __builtin_bswap64(v);
#endif
        return v;
    }
    v = 0;
    for (size_t i = 0; i < 8; i++) {
        uint64_t b = (byte + i < nbytes) ? buf[byte + i] : 0;
        v = (v << 8) | b;
    }
    return v;
}

static inline void br_load(br_t* r, size_t byte) {
    r->win = br_load64(r->buf, r->nbytes, byte);
    r->wnext = (byte + 8 < r->nbytes) ? r->buf[byte + 8] : 0;
    r->winbyte = byte;
}

static inline uint64_t br_mask(int nbits) {
    return (nbits >= 64) ? ~0ULL : ((1ULL << nbits) - 1ULL);
}

static inline uint64_t br_get(br_t* r, int nbits) {
    if (nbits <= 0) return 0;
    if (nbits > 64) {
        /* The two decoders never ask for more than 64 bits; keep the original
           byte-at-a-time path for completeness. */
        uint64_t v = 0;
        int got = 0;
        while (got < nbits) {
            size_t byte = r->bitpos >> 3;
            int off = (int)(r->bitpos & 7);
            int avail = 8 - off;
            int take = (nbits - got) < avail ? (nbits - got) : avail;
            uint64_t chunk = (uint64_t)(r->buf[byte] >> (avail - take));
            v = (v << take) | (chunk & ((1ULL << take) - 1ULL));
            got += take;
            r->bitpos += (size_t)take;
        }
        return v;
    }
    size_t byte = r->bitpos >> 3;
    int off = (int)(r->bitpos & 7);
    if (byte != r->winbyte) br_load(r, byte);
    uint64_t W = r->win;
    uint64_t v;
    if (off + nbits <= 64) {
        /* the whole field lies inside the window */
        v = (W >> (64 - off - nbits)) & br_mask(nbits);
    } else {
        /* the field crosses the window: its high part is the low (64 - off)
           bits of the window, its low part is the top k bits of the byte that
           follows the window */
        int k = nbits - (64 - off);
        uint64_t high = W & br_mask(64 - off);
        uint64_t low = ((uint64_t)r->wnext >> (8 - k)) & br_mask(k);
        v = (high << k) | low;
    }
    r->bitpos += (size_t)nbits;
    return v;
}

/* ------------------------------------------------------------------ Chimp128 */
#define PREV_VALUES 128
#define PREV_LOG2   7
#define THRESHOLD   (6 + PREV_LOG2)
#define SET_LSB     ((1 << (THRESHOLD + 1)) - 1)
#define FLAG_ZERO_SIZE (PREV_LOG2 + 2)
#define FLAG_ONE_SIZE  (PREV_LOG2 + 11)
#define INITIAL_FILL   (PREV_LOG2 + 9)

static inline uint64_t d2l(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static inline double l2d(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }

/* encode: returns bit count; bytes written to out */
long chimp128_encode(const double* vals, long n, uint8_t* out, long out_cap_bytes) {
    int32_t indices[1 << (THRESHOLD + 1)];
    uint64_t stored_values[PREV_VALUES];
    bw_t w;
    long index = 0, size = 0;
    int current = 0, stored_leading_zeros = 0x7FFFFFFF;
    long i;

    memset(indices, 0, sizeof(indices));
    memset(stored_values, 0, sizeof(stored_values));
    bw_init(&w, out, (size_t)out_cap_bytes);

    for (i = 0; i < n; i++) {
        uint64_t value = d2l(vals[i]);
        if (i == 0) {
            stored_values[current] = value;
            bw_put(&w, value, 64);
            indices[(int)((value & 0xFFFFFFFFULL) & (uint64_t)SET_LSB)] = (int32_t)index;
            size += 64;
        } else {
            int key = (int)((value & 0xFFFFFFFFULL) & (uint64_t)SET_LSB);
            int curr_index = indices[key];
            int trailing_zeros = 0;
            int previous_index;
            uint64_t xor;

            if ((index - curr_index) < PREV_VALUES) {
                uint64_t temp_xor = value ^ stored_values[curr_index % PREV_VALUES];
                trailing_zeros = ntz64(temp_xor);
                if (trailing_zeros > THRESHOLD) {
                    previous_index = curr_index % PREV_VALUES;
                    xor = temp_xor;
                } else {
                    previous_index = (int)(index % PREV_VALUES);
                    xor = stored_values[previous_index] ^ value;
                }
            } else {
                previous_index = (int)(index % PREV_VALUES);
                xor = stored_values[previous_index] ^ value;
            }

            if (xor == 0) {
                bw_put(&w, (uint64_t)previous_index, FLAG_ZERO_SIZE);
                size += FLAG_ZERO_SIZE;
                stored_leading_zeros = 65;
            } else {
                int leading_zeros = LEADING_ROUND[nlz64(xor)];
                if (trailing_zeros > THRESHOLD) {
                    int significant_bits = 64 - leading_zeros - trailing_zeros;
                    uint64_t meta = 512ULL * (uint64_t)(PREV_VALUES + previous_index)
                                  + 64ULL * (uint64_t)LEADING_REPRESENTATION[leading_zeros]
                                  + (uint64_t)significant_bits;
                    bw_put(&w, meta, FLAG_ONE_SIZE);
                    bw_put(&w, (xor >> trailing_zeros) & MASK64, significant_bits);
                    size += significant_bits + FLAG_ONE_SIZE;
                    stored_leading_zeros = 65;
                } else if (leading_zeros == stored_leading_zeros) {
                    int significant_bits = 64 - leading_zeros;
                    bw_put(&w, 2, 2);
                    bw_put(&w, xor & MASK64, significant_bits);
                    size += 2 + significant_bits;
                } else {
                    stored_leading_zeros = leading_zeros;
                    int significant_bits = 64 - leading_zeros;
                    bw_put(&w, (uint64_t)(24 + LEADING_REPRESENTATION[leading_zeros]), 5);
                    bw_put(&w, xor & MASK64, significant_bits);
                    size += 5 + significant_bits;
                }
            }
            current = (current + 1) % PREV_VALUES;
            stored_values[current] = value;
            index++;
            indices[key] = (int32_t)index;
        }
    }
    /* close(): encode NaN, then a terminating 0 bit */
    {
        uint64_t nanv = NAN_LONG;
        int key = (int)((nanv & 0xFFFFFFFFULL) & (uint64_t)SET_LSB);
        int curr_index = indices[key];
        int trailing_zeros = 0, previous_index;
        uint64_t xor;
        if ((index - curr_index) < PREV_VALUES) {
            uint64_t temp_xor = nanv ^ stored_values[curr_index % PREV_VALUES];
            trailing_zeros = ntz64(temp_xor);
            if (trailing_zeros > THRESHOLD) {
                previous_index = curr_index % PREV_VALUES;
                xor = temp_xor;
            } else {
                previous_index = (int)(index % PREV_VALUES);
                xor = stored_values[previous_index] ^ nanv;
            }
        } else {
            previous_index = (int)(index % PREV_VALUES);
            xor = stored_values[previous_index] ^ nanv;
        }
        if (xor == 0) {
            bw_put(&w, (uint64_t)previous_index, FLAG_ZERO_SIZE);
            size += FLAG_ZERO_SIZE;
        } else {
            int leading_zeros = LEADING_ROUND[nlz64(xor)];
            if (trailing_zeros > THRESHOLD) {
                int significant_bits = 64 - leading_zeros - trailing_zeros;
                uint64_t meta = 512ULL * (uint64_t)(PREV_VALUES + previous_index)
                              + 64ULL * (uint64_t)LEADING_REPRESENTATION[leading_zeros]
                              + (uint64_t)significant_bits;
                bw_put(&w, meta, FLAG_ONE_SIZE);
                bw_put(&w, (xor >> trailing_zeros) & MASK64, significant_bits);
                size += significant_bits + FLAG_ONE_SIZE;
            } else if (leading_zeros == stored_leading_zeros) {
                int significant_bits = 64 - leading_zeros;
                bw_put(&w, 2, 2);
                bw_put(&w, xor & MASK64, significant_bits);
                size += 2 + significant_bits;
            } else {
                stored_leading_zeros = leading_zeros;
                int significant_bits = 64 - leading_zeros;
                bw_put(&w, (uint64_t)(24 + LEADING_REPRESENTATION[leading_zeros]), 5);
                bw_put(&w, xor & MASK64, significant_bits);
                size += 5 + significant_bits;
            }
        }
        bw_put(&w, 0, 1);
        size += 1;
    }
    bw_flush(&w);
    return size;
}

/* decode: returns number of values written to out */
long chimp128_decode(const uint8_t* in, long in_bytes, double* out, long max_out) {
    br_t r;
    uint64_t stored_values[PREV_VALUES];
    long n = 0, i;
    int current = 0, stored_leading_zeros = 0x7FFFFFFF;
    uint64_t stored_val;

    memset(stored_values, 0, sizeof(stored_values));
    r.buf = in; r.nbytes = (size_t)in_bytes; r.bitpos = 0;
    r.win = 0; r.wnext = 0; r.winbyte = (size_t)-1;

    stored_val = br_get(&r, 64);
    if (stored_val == NAN_LONG) return 0;
    stored_values[current] = stored_val;
    out[n++] = l2d(stored_val);

    while (n < max_out) {
        int flag = (int)br_get(&r, 2);
        uint64_t value;
        if (flag == 3) {
            stored_leading_zeros = LEADING_FROM_CODE[(int)br_get(&r, 3)];
            value = br_get(&r, 64 - stored_leading_zeros);
            stored_val = stored_val ^ value;
        } else if (flag == 2) {
            value = br_get(&r, 64 - stored_leading_zeros);
            stored_val = stored_val ^ value;
        } else if (flag == 1) {
            uint64_t temp = br_get(&r, INITIAL_FILL);
            int index = (int)((temp >> (INITIAL_FILL - PREV_LOG2)) & ((1ULL << PREV_LOG2) - 1ULL));
            int leading = LEADING_FROM_CODE[(int)((temp >> 6) & 7ULL)];
            int significant_bits = (int)(temp & 63ULL);
            int stored_trailing_zeros;
            if (significant_bits == 0) significant_bits = 64;
            stored_leading_zeros = leading;
            stored_trailing_zeros = 64 - significant_bits - stored_leading_zeros;
            stored_val = stored_values[index];
            value = br_get(&r, 64 - stored_leading_zeros - stored_trailing_zeros);
            value = (value << stored_trailing_zeros) & MASK64;
            stored_val = stored_val ^ value;
        } else {
            stored_val = stored_values[br_get(&r, PREV_LOG2)];
        }
        if (stored_val == NAN_LONG) break;
        out[n++] = l2d(stored_val);
        current = (current + 1) % PREV_VALUES;
        stored_values[current] = stored_val;
    }
    (void)i;
    return n;
}

/* ------------------------------------------------------------------ Gorilla */
long gorilla_encode(const double* vals, long n, uint8_t* out, long out_cap_bytes) {
    bw_t w;
    long size = 0, i;
    uint64_t stored_val = 0;
    int stored_leading_zeros = 0x7FFFFFFF, stored_trailing_zeros = 0;

    bw_init(&w, out, (size_t)out_cap_bytes);
    for (i = 0; i < n; i++) {
        uint64_t v = d2l(vals[i]);
        if (i == 0) {
            stored_val = v;
            bw_put(&w, v, 64);
            size += 64;
            continue;
        }
        {
            uint64_t xor = stored_val ^ v;
            if (xor == 0) {
                bw_put(&w, 0, 1);
                size += 1;
            } else {
                int leading_zeros = nlz64(xor);
                int trailing_zeros = ntz64(xor);
                if (leading_zeros >= 32) leading_zeros = 31;
                bw_put(&w, 1, 1);
                size += 1;
                if (leading_zeros >= stored_leading_zeros
                        && trailing_zeros >= stored_trailing_zeros) {
                    int significant_bits = 64 - stored_leading_zeros - stored_trailing_zeros;
                    bw_put(&w, 0, 1);
                    bw_put(&w, (xor >> stored_trailing_zeros) & MASK64, significant_bits);
                    size += 1 + significant_bits;
                } else {
                    int significant_bits = 64 - leading_zeros - trailing_zeros;
                    bw_put(&w, 1, 1);
                    bw_put(&w, (uint64_t)leading_zeros, 5);
                    bw_put(&w, significant_bits == 64 ? 0ULL : (uint64_t)significant_bits, 6);
                    bw_put(&w, (xor >> trailing_zeros) & MASK64, significant_bits);
                    stored_leading_zeros = leading_zeros;
                    stored_trailing_zeros = trailing_zeros;
                    size += 1 + 5 + 6 + significant_bits;
                }
            }
            stored_val = v;
        }
    }
    {   /* close(): encode NaN then a terminating 0 bit */
        uint64_t v = NAN_LONG;
        uint64_t xor = stored_val ^ v;
        if (xor == 0) {
            bw_put(&w, 0, 1); size += 1;
        } else {
            int leading_zeros = nlz64(xor);
            int trailing_zeros = ntz64(xor);
            if (leading_zeros >= 32) leading_zeros = 31;
            bw_put(&w, 1, 1); size += 1;
            if (leading_zeros >= stored_leading_zeros && trailing_zeros >= stored_trailing_zeros) {
                int significant_bits = 64 - stored_leading_zeros - stored_trailing_zeros;
                bw_put(&w, 0, 1);
                bw_put(&w, (xor >> stored_trailing_zeros) & MASK64, significant_bits);
                size += 1 + significant_bits;
            } else {
                int significant_bits = 64 - leading_zeros - trailing_zeros;
                bw_put(&w, 1, 1);
                bw_put(&w, (uint64_t)leading_zeros, 5);
                bw_put(&w, significant_bits == 64 ? 0ULL : (uint64_t)significant_bits, 6);
                bw_put(&w, (xor >> trailing_zeros) & MASK64, significant_bits);
                size += 1 + 5 + 6 + significant_bits;
            }
        }
        bw_put(&w, 0, 1);
        size += 1;
    }
    bw_flush(&w);
    return size;
}

long gorilla_decode(const uint8_t* in, long in_bytes, double* out, long max_out) {
    br_t r;
    long n = 0;
    uint64_t stored_val;
    int stored_leading_zeros = 0x7FFFFFFF, stored_trailing_zeros = 0;

    r.buf = in; r.nbytes = (size_t)in_bytes; r.bitpos = 0;
    r.win = 0; r.wnext = 0; r.winbyte = (size_t)-1;
    stored_val = br_get(&r, 64);
    if (stored_val == NAN_LONG) return 0;
    out[n++] = l2d(stored_val);
    while (n < max_out) {
        uint64_t cur;
        if (br_get(&r, 1) == 0) {
            cur = stored_val;
        } else {
            uint64_t value;
            if (br_get(&r, 1) == 1) {
                /* ORDER MATTERS: the official Decompressor reads the 5-bit leading
                   zero count FIRST, then the 6-bit significant length. */
                stored_leading_zeros = (int)br_get(&r, 5);
                int significant = (int)br_get(&r, 6);
                if (significant == 0) significant = 64;
                stored_trailing_zeros = 64 - significant - stored_leading_zeros;
            }
            value = br_get(&r, 64 - stored_leading_zeros - stored_trailing_zeros);
            value = (value << stored_trailing_zeros) & MASK64;
            cur = stored_val ^ value;
        }
        if (cur == NAN_LONG) break;
        out[n++] = l2d(cur);
        stored_val = cur;
    }
    return n;
}

/* ------------------------------------------------------------------ batch decode */
/* Decode many segments that live in ONE contiguous arena in a single call.
   offs/lens are the byte range of every segment in the arena, vals is the number
   of values it holds, and idx lists the segments to decode (in order).  The
   results are written to out with a stride of `stride` doubles per segment.
   This pays the per-call overhead once per query instead of once per segment. */
static long decode_multi(const uint8_t* arena, const long* offs, const long* lens,
                          const long* vals, const long* idx, long n,
                          double* out, long stride, int which) {
    long total = 0;
    for (long i = 0; i < n; i++) {
        long j = idx[i];
        double* dst = out + i * stride;
        long got = (which == 0)
            ? chimp128_decode(arena + offs[j], lens[j], dst, vals[j])
            : gorilla_decode(arena + offs[j], lens[j], dst, vals[j]);
        total += got;
    }
    return total;
}

long chimp128_decode_multi(const uint8_t* arena, const long* offs, const long* lens,
                            const long* vals, const long* idx, long n,
                            double* out, long stride) {
    return decode_multi(arena, offs, lens, vals, idx, n, out, stride, 0);
}

long gorilla_decode_multi(const uint8_t* arena, const long* offs, const long* lens,
                          const long* vals, const long* idx, long n,
                          double* out, long stride) {
    return decode_multi(arena, offs, lens, vals, idx, n, out, stride, 1);
}

/* size probe: how many bytes does the encoder need for n values?  worst case bound */
long codec_max_bytes(long n) { return (long)(n * 12 + 64); }
