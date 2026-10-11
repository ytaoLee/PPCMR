"""Canonical commitments, owner signatures, and per-scope content encryption."""

import base64
import hashlib
import json
import os

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def encode(data):
    return base64.b64encode(data).decode("ascii")


def decode(data):
    return base64.b64decode(data, validate=True)


def sign(private_key, payload, purpose):
    message = purpose.encode("ascii") + b"\x00" + canonical(payload)
    return {"payload": payload, "signature": encode(private_key.sign(message))}


def verify_signature(public_key, envelope, purpose):
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "signature"}:
        raise ValueError("Malformed signed envelope")
    public_key.verify(decode(envelope["signature"]),
                      purpose.encode("ascii") + b"\x00" + canonical(envelope["payload"]))
    return envelope["payload"]


def merkle_root(records):
    """Root over ID-sorted metadata. Empty roots and leaves use domain separation.

    The baseline transmits ALL eligible metadata, not just membership paths.
    """
    ordered = sorted(records, key=lambda record: record["id"])
    if not ordered:
        return hashlib.sha256(b"ppcmr:empty:v1").hexdigest()
    level = [hashlib.sha256(b"ppcmr:leaf:v1\x00" + canonical(r)).digest()
             for r in ordered]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(b"ppcmr:node:v1\x00" + level[i] + level[i + 1]).digest()
                 for i in range(0, len(level), 2)]
    return level[0].hex()


def new_owner_key():
    return Ed25519PrivateKey.generate()


def export_public_key(public_key):
    return encode(public_key.public_bytes(serialization.Encoding.Raw,
                                          serialization.PublicFormat.Raw))


def load_public_key(encoded):
    return Ed25519PublicKey.from_public_bytes(decode(encoded))


def encrypt_content(key, plaintext, aad):
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), canonical(aad))
    return {"nonce": encode(nonce), "ciphertext": encode(ciphertext)}


def decrypt_content(key, blob, aad):
    return AESGCM(key).decrypt(decode(blob["nonce"]), decode(blob["ciphertext"]),
                               canonical(aad)).decode("utf-8")
