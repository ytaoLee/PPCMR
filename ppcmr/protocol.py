"""Owner / cloud / client protocol with full authorized-candidate witnesses.

The verifier proves exact top-k in the committed binary feature space. It does
not infer semantic relevance or provide query/index confidentiality.
"""

import copy
import os
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .crypto import (decrypt_content, digest, encrypt_content, export_public_key,
                     load_public_key, merkle_root, new_owner_key, sign, verify_signature)

MODALITIES = {"image", "text"}
RECORD_FIELDS = {"id", "scope", "modality", "hash_code", "content_digest"}
REQUEST_FIELDS = {"schema", "snapshot_digest", "subject", "query_modality",
                  "target_modality", "query_code", "k", "nonce", "signature"}


class ProtocolError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(condition, code, message):
    if not condition:
        raise ProtocolError(code, message)


def check_code(code, bits):
    require(isinstance(code, str) and len(code) == bits // 4 and
            all(c in "0123456789abcdef" for c in code),
            "MALFORMED_FEATURE", "Hash code must be fixed-length lowercase hexadecimal")


def aad_for(metadata, manifest):
    return {"database_id": manifest["database_id"], "epoch": manifest["epoch"],
            "id": metadata["id"], "scope": metadata["scope"],
            "modality": metadata["modality"], "hash_code": metadata["hash_code"]}


def similarity(query_code, item_code, bits):
    return bits - (int(query_code, 16) ^ int(item_code, 16)).bit_count()


def rank_candidates(records, request, bits):
    ranked = [{"id": r["id"], "score": similarity(request["query_code"], r["hash_code"], bits)}
              for r in records]
    # Deterministic tie breaking is part of the signed manifest.
    return sorted(ranked, key=lambda r: (-r["score"], r["id"]))[:request["k"]]


@dataclass
class ClientContext:
    """Delivered by the owner through a trusted channel, not by the cloud."""
    public_key: object
    trusted_manifest: dict
    credential: dict
    content_keys: dict
    query_signing_key: object = None


@dataclass
class Verification:
    accepted: bool
    code: str
    detail: str
    candidate_count: int = 0
    verified_ids: tuple = ()
    decrypted_contents: tuple = ()


class Owner:
    def __init__(self, database_id="ppcmr-demo"):
        self.database_id = database_id
        self.private_key = new_owner_key()

    @property
    def public_key(self):
        return self.private_key.public_key()

    def publish(self, records, grants, bits=64, epoch=1, permission_epoch=1,
                feature_method="synthetic_correlated_binary_v1"):
        """Build fresh content keys and a signed, versioned complete snapshot."""
        require(type(bits) is int and bits >= 8 and bits % 8 == 0,
                "BAD_CONFIG", "bits must be a positive multiple of 8")
        require(type(epoch) is int and epoch > 0 and type(permission_epoch) is int
                and permission_epoch > 0, "BAD_CONFIG", "Epochs must be positive integers")
        records = copy.deepcopy(records)
        require(isinstance(records, list) and all(isinstance(r, dict) for r in records),
                "BAD_DATASET", "records must be a list of objects")
        seen = set()
        scopes = set()
        for record in records:
            require(set(record) == {"id", "scope", "modality", "hash_code", "content"},
                    "BAD_DATASET", "Unexpected input record fields")
            require(isinstance(record["id"], str) and record["id"] and
                    isinstance(record["scope"], str) and record["scope"],
                    "BAD_DATASET", "IDs and scopes must be nonempty strings")
            require(record["id"] not in seen, "BAD_DATASET", "Duplicate record ID")
            require(record["modality"] in MODALITIES and isinstance(record["content"], str),
                    "BAD_DATASET", "Unsupported modality or content type")
            check_code(record["hash_code"], bits)
            seen.add(record["id"])
            scopes.add(record["scope"])
        require(isinstance(grants, dict) and all(isinstance(u, str) and u for u in grants),
                "BAD_GRANTS", "grants must map subjects to scope lists")
        for rights in grants.values():
            require(isinstance(rights, list) and all(isinstance(s, str) for s in rights)
                    and len(rights) == len(set(rights)) and set(rights) <= scopes,
                    "BAD_GRANTS", "Grant contains duplicate or unknown scopes")
        payload = {"schema": "ppcmr.snapshot.v1", "database_id": self.database_id,
                   "epoch": epoch, "permission_epoch": permission_epoch,
                   "hash_bits": bits, "feature_method": feature_method,
                   "rank_rule": "similarity_desc_then_id_asc", "partitions": []}
        keys = {scope: AESGCM.generate_key(bit_length=256) for scope in scopes}
        metadata, contents = [], {}
        for r in records:
            m = {field: r[field] for field in ("id", "scope", "modality", "hash_code")}
            blob = encrypt_content(keys[r["scope"]], r["content"], aad_for(m, payload))
            m["content_digest"] = digest(blob)
            metadata.append(m)
            contents[r["id"]] = blob
        for scope in sorted(scopes):
            for modality in sorted(MODALITIES):
                partition = [m for m in metadata if m["scope"] == scope and m["modality"] == modality]
                payload["partitions"].append({"scope": scope, "modality": modality,
                                               "count": len(partition), "root": merkle_root(partition)})
        manifest = sign(self.private_key, payload, "ppcmr.snapshot.v1")
        credentials = {}
        query_keys = {subject: new_owner_key() for subject in grants}
        for subject, rights in grants.items():
            grant = {"schema": "ppcmr.credential.v1", "subject": subject,
                     "snapshot_digest": digest(payload), "database_id": self.database_id,
                     "epoch": epoch, "permission_epoch": permission_epoch,
                     "allowed_scopes": sorted(rights),
                     "query_public_key": export_public_key(query_keys[subject].public_key())}
            credentials[subject] = sign(self.private_key, grant, "ppcmr.credential.v1")
        server = Cloud(self.public_key, manifest, metadata, contents)
        clients = {subject: ClientContext(self.public_key, copy.deepcopy(manifest),
                    copy.deepcopy(credentials[subject]), {s: keys[s] for s in rights}, query_keys[subject])
                   for subject, rights in grants.items()}
        return server, clients


def validated_state(public_key, trusted_manifest, credential, request):
    """Use a separately pinned owner key and the latest independently obtained state."""
    try:
        manifest = verify_signature(public_key, trusted_manifest, "ppcmr.snapshot.v1")
    except (InvalidSignature, ValueError, TypeError, KeyError):
        raise ProtocolError("BAD_MANIFEST_SIGNATURE", "Owner snapshot signature is invalid") from None
    require(manifest["schema"] == "ppcmr.snapshot.v1" and
            manifest["rank_rule"] == "similarity_desc_then_id_asc",
            "BAD_MANIFEST", "Unsupported snapshot schema or ranking rule")
    try:
        grant = verify_signature(public_key, credential, "ppcmr.credential.v1")
    except (InvalidSignature, ValueError, TypeError, KeyError):
        raise ProtocolError("BAD_CREDENTIAL_SIGNATURE", "Authorization signature is invalid") from None
    require(grant["schema"] == "ppcmr.credential.v1", "BAD_CREDENTIAL", "Unsupported credential")
    require(grant["snapshot_digest"] == digest(manifest) and
            grant["database_id"] == manifest["database_id"] and
            grant["epoch"] == manifest["epoch"] and
            grant["permission_epoch"] == manifest["permission_epoch"],
            "STALE_CREDENTIAL", "Credential does not belong to the current trusted snapshot")
    require(isinstance(request, dict) and set(request) == REQUEST_FIELDS and
            request["schema"] == "ppcmr.request.v1", "BAD_REQUEST", "Malformed request")
    require(request["snapshot_digest"] == digest(manifest),
            "STALE_REQUEST", "Query references a different snapshot")
    require(request["subject"] == grant["subject"], "WRONG_SUBJECT", "Credential belongs to another subject")
    try:
        request_payload = {key: value for key, value in request.items() if key != "signature"}
        verify_signature(load_public_key(grant["query_public_key"]),
                         {"payload": request_payload, "signature": request["signature"]},
                         "ppcmr.request.v1")
    except (InvalidSignature, ValueError, TypeError, KeyError):
        raise ProtocolError("BAD_REQUEST_SIGNATURE", "Query is not authenticated by the authorized client") from None
    require(request["query_modality"] in MODALITIES and request["target_modality"] in MODALITIES
            and request["query_modality"] != request["target_modality"],
            "BAD_REQUEST", "The query must cross image/text modalities")
    require(type(request["k"]) is int and request["k"] > 0,
            "BAD_REQUEST", "k must be a positive integer")
    require(isinstance(request["nonce"], str) and len(request["nonce"]) == 32 and
            all(c in "0123456789abcdef" for c in request["nonce"]),
            "BAD_REQUEST", "A 128-bit query nonce is required")
    check_code(request["query_code"], manifest["hash_bits"])
    scopes = {p["scope"] for p in manifest["partitions"]}
    rights = grant["allowed_scopes"]
    require(isinstance(rights, list) and len(rights) == len(set(rights)) and set(rights) <= scopes,
            "BAD_CREDENTIAL", "Credential contains unknown or duplicate scopes")
    partitions = [p for p in manifest["partitions"] if p["scope"] in rights
                  and p["modality"] == request["target_modality"]]
    return manifest, grant, partitions


def make_request(context, query_code, query_modality, k=5, nonce=None):
    payload = {"schema": "ppcmr.request.v1", "snapshot_digest": digest(context.trusted_manifest["payload"]),
            "subject": context.credential["payload"]["subject"], "query_modality": query_modality,
            "target_modality": "text" if query_modality == "image" else "image",
            "query_code": query_code, "k": k, "nonce": nonce or os.urandom(16).hex()}
    envelope = sign(context.query_signing_key, payload, "ppcmr.request.v1")
    return {**payload, "signature": envelope["signature"]}


class Cloud:
    """Contains ciphertext content and PUBLIC hash metadata; no content keys."""
    def __init__(self, public_key, manifest, metadata, contents):
        self.public_key = public_key
        self.manifest = copy.deepcopy(manifest)
        self.metadata = copy.deepcopy(metadata)
        self.contents = copy.deepcopy(contents)

    def response_for_ranked(self, request, credential, ranked, include_witnesses=True):
        manifest, _, partitions = validated_state(self.public_key, self.manifest, credential, request)
        witnesses = [{"scope": p["scope"], "modality": p["modality"],
                      "records": sorted([copy.deepcopy(m) for m in self.metadata
                                         if m["scope"] == p["scope"] and m["modality"] == p["modality"]],
                                        key=lambda m: m["id"])} for p in partitions] if include_witnesses else []
        return {"schema": "ppcmr.response.v1", "snapshot_digest": digest(manifest),
                "request_digest": digest(request), "witnesses": witnesses,
                "results": [{**r, "content": copy.deepcopy(self.contents[r["id"]])} for r in ranked]}

    def search(self, request, credential, include_witnesses=True):
        manifest, grant, _ = validated_state(self.public_key, self.manifest, credential, request)
        candidates = [m for m in self.metadata if m["scope"] in grant["allowed_scopes"]
                      and m["modality"] == request["target_modality"]]
        ranked = rank_candidates(candidates, request, manifest["hash_bits"])
        return self.response_for_ranked(request, credential, ranked, include_witnesses)


def verify_response(context, expected_request, response, decrypt=False):
    """Fail closed, even when cloud output is malformed.

    Context and expected_request are trusted client state. Neither is replaced
    with state supplied inside the cloud response.
    """
    try:
        manifest, _, partitions = validated_state(context.public_key, context.trusted_manifest,
                                                  context.credential, expected_request)
        require(isinstance(response, dict) and set(response) ==
                {"schema", "snapshot_digest", "request_digest", "witnesses", "results"} and
                response["schema"] == "ppcmr.response.v1", "BAD_RESPONSE", "Malformed response")
        require(response["snapshot_digest"] == digest(manifest),
                "STALE_RESPONSE", "Response uses a different snapshot")
        require(response["request_digest"] == digest(expected_request),
                "QUERY_BINDING_FAILED", "Response is not bound to this query and nonce")
        require(isinstance(response["witnesses"], list), "BAD_WITNESS", "Witnesses must be a list")
        expected = {(p["scope"], p["modality"]): p for p in partitions}
        witnesses = {}
        for witness in response["witnesses"]:
            require(isinstance(witness, dict) and set(witness) == {"scope", "modality", "records"},
                    "BAD_WITNESS", "Malformed partition witness")
            partition_id = (witness["scope"], witness["modality"])
            require(partition_id in expected, "UNAUTHORIZED_PARTITION", "Unexpected partition in proof")
            require(partition_id not in witnesses, "DUPLICATE_PARTITION", "Repeated partition in proof")
            witnesses[partition_id] = witness
        require(set(witnesses) == set(expected), "MISSING_PARTITION", "An authorized partition is missing")
        all_records, by_id = [], {}
        for partition_id, partition in expected.items():
            records = witnesses[partition_id]["records"]
            require(isinstance(records, list) and len(records) == partition["count"],
                    "CANDIDATE_COUNT_MISMATCH", "Authorized candidate count does not match snapshot")
            for m in records:
                require(isinstance(m, dict) and set(m) == RECORD_FIELDS,
                        "BAD_WITNESS", "Malformed record metadata")
                require((m["scope"], m["modality"]) == partition_id,
                        "WRONG_PARTITION", "Record placed in the wrong partition")
                require(isinstance(m["id"], str) and m["id"] and m["id"] not in by_id,
                        "DUPLICATE_RECORD", "Missing or repeated record ID")
                check_code(m["hash_code"], manifest["hash_bits"])
                require(isinstance(m["content_digest"], str) and len(m["content_digest"]) == 64,
                        "BAD_WITNESS", "Malformed content digest")
                by_id[m["id"]] = m
            require(merkle_root(records) == partition["root"],
                    "CANDIDATE_ROOT_MISMATCH", "Candidate metadata does not match owner commitment")
            all_records.extend(records)
        expected_ranked = rank_candidates(all_records, expected_request, manifest["hash_bits"])
        results = response["results"]
        require(isinstance(results, list) and len(results) == len(expected_ranked),
                "RESULT_COUNT_MISMATCH", "Return exactly min(k, authorized candidate count) records")
        for actual in results:
            require(isinstance(actual, dict) and set(actual) == {"id", "score", "content"}
                    and type(actual["score"]) is int,
                    "BAD_RESULT", "Malformed result or non-integer score")
            require(isinstance(actual["id"], str) and actual["id"] in by_id,
                    "UNAUTHORIZED_RESULT", "Result is absent from the authorized candidate set")
        actual_ranked = [{"id": r["id"], "score": r["score"]} for r in results]
        require(actual_ranked == expected_ranked, "TOPK_MISMATCH",
                "Result set, scores, or ordering differs from exact authorized top-k")
        plaintexts = []
        for result in results:
            m = by_id[result["id"]]
            blob = result["content"]
            require(isinstance(blob, dict) and set(blob) == {"nonce", "ciphertext"},
                    "BAD_CONTENT", "Malformed encrypted content")
            require(digest(blob) == m["content_digest"], "CONTENT_DIGEST_MISMATCH", "Returned ciphertext was modified")
            if decrypt:
                require(m["scope"] in context.content_keys, "MISSING_CONTENT_KEY", "Client has no scope key")
                try:
                    plaintexts.append(decrypt_content(context.content_keys[m["scope"]], blob,
                                                      aad_for(m, manifest)))
                except (InvalidTag, ValueError, TypeError, UnicodeDecodeError):
                    raise ProtocolError("CONTENT_DECRYPTION_FAILED", "Content authentication/decryption failed") from None
        return Verification(True, "ACCEPT", "Verified exact top-k over the complete authorized candidate set",
                            len(all_records), tuple(r["id"] for r in results), tuple(plaintexts))
    except ProtocolError as error:
        return Verification(False, error.code, str(error))
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return Verification(False, "MALFORMED_MESSAGE", "Malformed protocol message")
