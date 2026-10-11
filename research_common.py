"""Data identities, deterministic binary features and retrieval metrics."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts" / "real_coco"
DATA_ROOT = Path(os.environ.get("PPCMR_COCO_ROOT", str(BASE / "data" / "coco")))
MODEL = Path(os.environ.get("PPCMR_CLIP_WEIGHTS", str(BASE / "weights" / "open_clip_model.safetensors")))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def value_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def projection(dim, bits, seed):
    # All code lengths are prefixes of one fixed 512-column draw.
    return np.random.default_rng(seed).standard_normal((dim, 512), dtype=np.float32)[:, :bits]


def binary_features(features, bits, seed):
    return np.where(features @ projection(features.shape[1], bits, seed) >= 0, 1, -1).astype(np.float32)


def hex_codes(features):
    return [row.tobytes().hex() for row in np.packbits(features > 0, axis=1, bitorder="big")]


def load_features(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads((folder / "encoder.json").read_text(encoding="utf-8"))
    if file_sha256(folder / "manifest.json") != provenance["manifest_sha256"]:
        raise ValueError("Manifest changed after feature encoding")
    if file_sha256(folder / "features.npz") != provenance["features_sha256"]:
        raise ValueError("Feature cache checksum mismatch")
    with np.load(folder / "features.npz", allow_pickle=False) as cache:
        images, texts = cache["images"], cache["texts"]
    if len(images) != len(manifest["images"]) or len(texts) != len(images) * 5:
        raise ValueError("Feature rows are not aligned with the manifest")
    if not np.isfinite(images).all() or not np.isfinite(texts).all():
        raise ValueError("Non-finite features")
    return manifest, provenance, images, texts


def entity_arrays(manifest):
    image_ids = np.array([f"image-{x['image_id']:012d}" for x in manifest["images"]])
    text_ids = np.array([f"text-{c['caption_id']:012d}" for x in manifest["images"] for c in x["captions"]])
    image_labels = np.arange(len(image_ids))
    text_labels = np.repeat(image_labels, 5)
    scopes = np.array([x["scope"] for x in manifest["images"]])
    return image_ids, text_ids, image_labels, text_labels, scopes, np.repeat(scopes, 5)


def measure_retrieval(queries, candidates, query_labels, candidate_labels, candidate_ids, chunk=64):
    """Exact relevant-item ranks, with lexicographic ID tie breaking.

    Avoid sorting full galleries: each positive rank is obtained by counting
    higher scores and equal-score predecessors. AP uses all paired positives.
    """
    order = np.argsort(candidate_ids, kind="stable")
    candidates, labels = candidates[order], candidate_labels[order]
    positions = {int(label): np.flatnonzero(labels == label) for label in np.unique(labels)}
    best, aps, positive_recall, best_ties = [], [], [], []
    for start in range(0, len(queries), chunk):
        scores = queries[start:start + chunk] @ candidates.T
        for i, label in enumerate(query_labels[start:start + chunk]):
            positive = positions.get(int(label))
            if positive is None:
                raise ValueError("A query has no authorized positive; do not mix it into conditional recall")
            s = scores[i]
            ranks = np.array([np.count_nonzero(s > s[p]) + np.count_nonzero(s[:p] == s[p]) for p in positive])
            ranks.sort()
            best.append(int(ranks[0]) + 1)
            aps.append(float(np.mean(np.arange(1, len(ranks) + 1) / (ranks + 1))))
            positive_recall.append([float(np.mean(ranks < k)) for k in (1, 5, 10)])
            best_ties.append(int(np.count_nonzero(s == np.max(s))))
    best = np.asarray(best)
    metrics = {"queries": len(queries), "candidates": len(candidates),
               **{f"R@{k}": float(np.mean(best <= k)) for k in (1, 5, 10)},
               "mAP": float(np.mean(aps)), "median_rank": float(np.median(best)),
               "mean_rank": float(np.mean(best)),
               "top_score_tie_query_fraction": float(np.mean(np.asarray(best_ties) > 1)),
               **{f"positive_recall@{k}": float(np.mean(positive_recall, axis=0)[i]) for i, k in enumerate((1, 5, 10))}}
    return metrics, {"best_rank": best, "ap": np.asarray(aps)}
