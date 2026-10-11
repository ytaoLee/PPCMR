"""Reproducible synthetic paired binary features; no trained image/text encoder."""

import random


def synthetic_records(pairs=500, bits=64, scopes=10, seed=20261007):
    if type(pairs) is not int or pairs < 1:
        raise ValueError("pairs must be a positive integer")
    if type(bits) is not int or bits < 8 or bits % 8:
        raise ValueError("bits must be a positive multiple of 8")
    if type(scopes) is not int or scopes < 1:
        raise ValueError("scopes must be a positive integer")
    rng = random.Random(seed)
    records = []
    for i in range(pairs):
        latent = rng.getrandbits(bits)
        scope = f"project-{i % scopes:02d}"
        for modality in ("image", "text"):
            code = latent
            for position in rng.sample(range(bits), max(1, bits // 32)):
                code ^= 1 << position
            records.append({"id": f"{modality}-{i:06d}", "scope": scope,
                            "modality": modality, "hash_code": f"{code:0{bits // 4}x}",
                            "content": f"Synthetic {modality} item, pair {i}; no real media."})
    return records
