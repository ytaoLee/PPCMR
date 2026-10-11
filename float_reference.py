"""Exact mathematical inner products of the original FP32 feature VALUES.

No encryption is performed here. Every finite binary32 value is an integer
multiple of 2**-149. This encoding does not quantize features. Signed zeros
have the same mathematical value. Exact sums differ from a BLAS implementation
that rounds intermediate products/additions to FP32; that distinction must be
reported rather than hidden by changing the reference.
"""

import numpy as np

SCALE_EXPONENT = 149
SCORE_SCALE_EXPONENT = 298


def encode_fp32(vector):
    vector = np.asarray(vector)
    if vector.dtype != np.float32 or vector.ndim != 1:
        raise ValueError("Expected a one-dimensional FP32 vector; implicit conversion is forbidden")
    if not np.isfinite(vector).all():
        raise ValueError("NaN and infinity are outside the retrieval reference")
    bits = np.ascontiguousarray(vector).view(np.uint32)
    values = []
    for word in bits:
        word = int(word)
        exponent, mantissa = (word >> 23) & 255, word & 0x7fffff
        value = mantissa if exponent == 0 else (mantissa | 0x800000) << (exponent - 1)
        values.append(-value if word >> 31 else value)
    return tuple(values)


def exact_dot_encoded(query, candidate):
    if len(query) != len(candidate):
        raise ValueError("Dimension mismatch")
    return sum(x * y for x, y in zip(query, candidate))


def exact_dot(query, candidate):
    return exact_dot_encoded(encode_fp32(query), encode_fp32(candidate))


def score_as_float(score):
    return float(score) * (2.0 ** -SCORE_SCALE_EXPONENT)


def certified_topk(query, candidates, scores64, groups, k=10):
    """Certify the exact top-k using FP64 intervals; exact integers resolve ties.

    candidates must already be in ascending ID order. groups identify identical
    FP32 row bytes. The conservative FP64 error bound covers any ordinary
    reduction order with at most dimension products/additions (no overflow or
    underflow for products of finite FP32 values in FP64).
    """
    if not 1 <= k <= len(candidates) or len(scores64) != len(candidates):
        raise ValueError("Invalid candidate count or k")
    dim = candidates.shape[1]
    if query.dtype != np.float32 or candidates.dtype != np.float32:
        raise ValueError("Reference features must be FP32")
    if not np.isfinite(scores64).all():
        raise ValueError("Non-finite FP64 screening scores")
    # epsilon is twice the unit roundoff, deliberately conservative. Use a
    # maximum-component bound, avoiding a rounded norm as a proof premise.
    eps = np.finfo(np.float64).eps
    gamma = dim * eps / (1.0 - dim * eps)
    product_bound = dim * float(np.max(np.abs(query))) * float(np.max(np.abs(candidates)))
    error = np.nextafter(gamma * product_bound * (1.0 + 1e-12), np.inf)
    threshold = np.partition(scores64, len(scores64) - k)[len(scores64) - k]
    pool = np.flatnonzero(scores64 >= threshold - 2.0 * error)
    ordered = pool[np.argsort(-scores64[pool], kind="stable")]
    output = []
    integer_rows = 0
    encoded_query = None
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and scores64[ordered[end-1]] - scores64[ordered[end]] <= 2.0 * error:
            end += 1
        cluster = ordered[start:end].tolist()
        if len(cluster) > 1:
            if len({int(groups[i]) for i in cluster}) == 1:
                cluster.sort()  # Identical feature values: exact tie, ID rule.
            else:
                if encoded_query is None:
                    encoded_query = encode_fp32(query)
                exact = {i: exact_dot_encoded(encoded_query, encode_fp32(candidates[i])) for i in cluster}
                cluster.sort(key=lambda i: (-exact[i], i))
                integer_rows += len(cluster)
        output.extend(cluster)
        start = end
        if len(output) >= k:
            break
    return np.asarray(output[:k], dtype=np.int64), integer_rows, float(error)
