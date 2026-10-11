"""Compare existing batched FP32 CLIP rankings to a certified exact reference.

This is a plaintext arithmetic audit, not a private retrieval experiment.
FP64 screening with conservative error intervals certifies ranks; ambiguous
different embeddings are compared using lossless integer encodings.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import hashlib
from fractions import Fraction
from pathlib import Path
from time import perf_counter

import numpy as np

from float_reference import certified_topk, encode_fp32, exact_dot, score_as_float
from research_common import ARTIFACTS, entity_arrays, load_features, write_json


def feature_groups(candidates):
    lookup = {}
    groups = []
    for row in candidates:
        fingerprint = hashlib.sha256(row.tobytes()).digest()
        groups.append(lookup.setdefault(fingerprint, len(lookup)))
    return np.asarray(groups, dtype=np.int64)


def self_check():
    rng = np.random.default_rng(20261008)
    words = rng.integers(0, 2**32, 1500, dtype=np.uint32)
    words = words[((words >> 23) & 255) != 255]
    words = np.r_[words, np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff,
                                   0x00800000, 0x3f800000, 0x7f7fffff], dtype=np.uint32)]
    values = words.view(np.float32)
    encoded = encode_fp32(values)
    for x, n in zip(values, encoded):
        assert Fraction(float(x)) == Fraction(n, 2**149)
    # A strict exact ranking can become a rounded FP32 tie.
    q = np.array([1., 2.**-12], dtype=np.float32)
    c = np.array([[1., 0.], [1., 2.**-12]], dtype=np.float32)
    assert c[0] @ q == c[1] @ q
    assert exact_dot(q, c[1]) > exact_dot(q, c[0])
    # Compare the certified result with independent Fraction sums for random
    # rows, duplicates, zeros, and a perturbation below FP64's dot resolution.
    for dim in (2, 7, 512):
        query = rng.uniform(-1, 1, dim).astype(np.float32)
        candidates = rng.uniform(-1, 1, (23, dim)).astype(np.float32)
        candidates[4] = candidates[3]
        scores = candidates.astype(np.float64) @ query.astype(np.float64)
        result, _, _ = certified_topk(query, candidates, scores, feature_groups(candidates), 10)
        reference = [sum(Fraction(float(a))*Fraction(float(b)) for a,b in zip(query,row)) for row in candidates]
        expected = sorted(range(len(candidates)), key=lambda i: (-reference[i], i))[:10]
        assert result.tolist() == expected
    query = np.array([1, 2**-30], dtype=np.float32)
    candidates = np.array([[1, 0], [1, 2**-30], [1, 0], [0, 0]], dtype=np.float32)
    result, count, _ = certified_topk(query, candidates,
            candidates.astype(np.float64) @ query.astype(np.float64), feature_groups(candidates), 3)
    assert result.tolist() == [1, 0, 2] and count == 3
    zeros = np.zeros((12, 512), dtype=np.float32)
    result, _, _ = certified_topk(zeros[0], zeros, np.zeros(12), feature_groups(zeros), 10)
    assert result.tolist() == list(range(10))
    return {"finite_fp32_values_checked": len(values),
            "independent_fraction_rank_checks": 3,
            "sub_fp64_gap_exact_resolution": "passed",
            "duplicate_and_zero_id_ties": "passed",
            "fp32_rounding_counterexample": {
                "query": q.tolist(), "candidates": c.tolist(),
                "fp32_scores": (c @ q).tolist(),
                "exact_scores": [str(Fraction(exact_dot(q, row), 2**298)) for row in c],
                "fp32_id_order": [0, 1], "exact_order": [1, 0]}}


def audit_direction(direction, queries, candidates, query_labels, candidate_labels, ids, chunk=64):
    order = np.argsort(ids, kind="stable")
    candidates = candidates[order]
    candidate_labels, ids = candidate_labels[order], ids[order]
    groups = feature_groups(candidates)
    c64 = candidates.astype(np.float64)
    matches = {k: 0 for k in (1, 5, 10)}
    hits32 = {k: 0 for k in matches}
    hits_exact = {k: 0 for k in matches}
    integer_rows = 0
    mismatches = []
    exact_top10 = []
    max_bound = 0.0
    started = perf_counter()
    last_update = started
    for begin in range(0, len(queries), chunk):
        block = queries[begin:begin+chunk]
        scores32 = block @ candidates.T  # Same batch size/order as existing baseline.
        scores64 = block.astype(np.float64) @ c64.T
        top32 = np.argsort(-scores32, axis=1, kind="stable")[:, :10]
        for j, query in enumerate(block):
            exact, fallback, bound = certified_topk(query, candidates, scores64[j], groups, 10)
            integer_rows += fallback
            max_bound = max(max_bound, bound)
            exact_top10.append(ids[exact].tolist())
            label = query_labels[begin+j]
            for k in matches:
                matches[k] += int(np.array_equal(top32[j, :k], exact[:k]))
                hits32[k] += int(np.any(candidate_labels[top32[j, :k]] == label))
                hits_exact[k] += int(np.any(candidate_labels[exact[:k]] == label))
            if not np.array_equal(top32[j], exact):
                mismatches.append({"query_index": begin+j,
                    "fp32_ids": ids[top32[j]].tolist(), "exact_ids": ids[exact].tolist(),
                    "fp32_scores": scores32[j, top32[j]].astype(float).tolist(),
                    "fp64_scores": scores64[j, exact].tolist(),
                    "fp64_error_bound": bound, "exact_integer_rows_resolved": fallback})
        if perf_counter() - last_update >= 10.0:
            print(direction, min(begin+chunk, len(queries)), "/", len(queries), flush=True)
            last_update = perf_counter()
    return {"queries": len(queries), "candidates": len(candidates),
            "seconds": perf_counter() - started,
            "topk_order_equal_queries": {str(k): v for k,v in matches.items()},
            "fp32_recall": {f"R@{k}": hits32[k]/len(queries) for k in matches},
            "exact_recall": {f"R@{k}": hits_exact[k]/len(queries) for k in matches},
            "recall_delta_exact_minus_fp32": {f"R@{k}": (hits_exact[k]-hits32[k])/len(queries) for k in matches},
            "exact_integer_rows_resolved": integer_rows,
            "largest_fp64_screening_error_bound": max_bound,
            "mismatches": mismatches}, exact_top10


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=Path, default=ARTIFACTS / "test")
    p.add_argument("--output", type=Path, default=ARTIFACTS / "float_reference")
    args = p.parse_args()
    checks = self_check()
    manifest, provenance, images, texts = load_features(args.folder)
    ii, ti, il, tl, _, _ = entity_arrays(manifest)
    report = {"schema": "ppcmr.plaintext_float_arithmetic_audit.v1",
              "privacy_computation_performed": False, "self_checks": checks,
              "model": provenance["model"], "dimensions": images.shape[1],
              "dataset_split": manifest["split"], "folder": str(args.folder.resolve()),
              "manifest_sha256": provenance["manifest_sha256"],
              "features_sha256": provenance["features_sha256"],
              "fp32_reference": "NumPy batched FP32 dot, chunk=64, ID-ascending candidate order",
              "exact_reference": "exact mathematical inner product of unchanged FP32 values, integer scale 2^149",
              "ranking_rule": "score descending, record ID ascending for EXACT score ties",
              "certification": "conservative FP64 error intervals; exact integer comparisons for ambiguous different vectors",
              "directions": {}}
    rankings = {}
    for direction, q, c, ql, cl, ids in (
        ("image-to-text", images, texts, il, tl, ti),
        ("text-to-image", texts, images, tl, il, ii)):
        details, top = audit_direction(direction, q, c, ql, cl, ids)
        report["directions"][direction] = details
        rankings[direction] = top
        print(direction, {k:v for k,v in details.items() if k != "mismatches"}, flush=True)
    write_json(args.output / "arithmetic_audit.json", report)
    write_json(args.output / "exact_top10_ids.json", rankings)
    print("Saved", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
