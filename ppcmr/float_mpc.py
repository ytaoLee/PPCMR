"""Two-process additive-sharing retrieval with exact FP32-value inner products.

Research baseline: two non-colluding semi-honest compute parties; a trusted
dealer supplies FRESH matrix/vector Beaver triples. The client reconstructs
all authorized scores, selects top-k, and verifies full encrypted-feature
witnesses. This is O(N) client verification, not a succinct cloud-top-k proof.
Local multiprocessing pipes simulate private channels, not a remote deployment.
"""

import hashlib
import json
import multiprocessing as mp
import os
import pickle
import struct
import time
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from float_reference import encode_fp32, exact_dot_encoded, score_as_float
from .crypto import (canonical, decode, decrypt_content, digest, encode,
    encrypt_content, export_public_key, load_public_key, merkle_root,
    new_owner_key, sign, verify_signature)
from .publication_policy import (FEATURE_PROFILE, POLICY, SKETCH_PADDED_BYTES,
                                 SNAPSHOT_SCHEMA, validate_publication)

DIM = 512
BITS = 320
WORD_BYTES = BITS // 8
MODULUS = 1 << BITS
MASK = MODULUS - 1
HALF = MODULUS >> 1
MODALITIES = {"image", "text"}
FIELDS = {"id", "scope", "modality", "feature", "content_digest"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_feature(feature):
    require(isinstance(feature, np.ndarray) and feature.dtype == np.float32 and
            feature.shape == (DIM,), "Expected an unchanged 512-d FP32 vector")
    require(np.isfinite(feature).all() and np.max(np.abs(feature)) <= 1.0,
            "Feature components must be finite and within [-1,1]")
    # |score_integer| <= 512*2**298 = 2**307 < 2**319: no wraparound.
    return encode_fp32(feature)


def random_words(count):
    data = os.urandom(count * WORD_BYTES)
    return [int.from_bytes(data[i:i+WORD_BYTES], "little")
            for i in range(0, len(data), WORD_BYTES)]


def pack_words(words):
    require(all(type(x) is int and 0 <= x <= MASK for x in words), "Invalid ring shares")
    return b"".join(x.to_bytes(WORD_BYTES, "little") for x in words)


def share_digest(words):
    return hashlib.sha256(pack_words(words)).hexdigest()


def split_values(values):
    first = random_words(len(values))
    return first, [(x-a) & MASK for x,a in zip(values, first)]


def make_triple(rows):
    """Trusted preprocessing of [A],[b],[A b], independent of feature inputs."""
    if rows == 0:
        empty = {"id": os.urandom(16).hex(), "A": [], "b": [], "c": []}
        return empty, empty.copy()
    flat_a, b = random_words(rows * DIM), random_words(DIM)
    a = [flat_a[i:i+DIM] for i in range(0, len(flat_a), DIM)]
    c = [sum(x*y for x,y in zip(row,b)) & MASK for row in a]
    a0, a1 = split_values(flat_a)
    b0, b1 = split_values(b)
    c0, c1 = split_values(c)
    identifier = os.urandom(16).hex()
    return tuple({"id": identifier, "A": [aa[i:i+DIM] for i in range(0, len(aa), DIM)],
                  "b": bb, "c": cc} for aa,bb,cc in ((a0,b0,c0),(a1,b1,c1)))


def aad(snapshot, record, purpose):
    return {"database": snapshot["database"], "epoch": snapshot["epoch"],
            "id": record["id"], "scope": record["scope"],
            "modality": record["modality"], "purpose": purpose}


@dataclass
class FloatClient:
    owner_public: str
    snapshot: dict
    credential: dict
    scope_keys: dict
    signing_key: object
    verification_sketches: dict = None


@dataclass
class QuerySession:
    request: dict
    feature: np.ndarray
    shares: tuple
    share_envelopes: tuple


def publish(records, grants, provenance, epoch=1, database="coco-float-mpc", owner_key=None, compact=False):
    """Publish protected records, excluding private research provenance.

    The provenance argument is retained for runner compatibility; callers
    archive it in their experiment reports, never in server-visible objects.
    The public profile describes accepted values, not their encoder origin.
    """
    require(records and epoch >= 1, "Nonempty database and positive epoch required")
    owner = new_owner_key() if owner_key is None else owner_key
    owner_public = export_public_key(owner.public_key())
    scopes = sorted({r["scope"] for r in records})
    keys = {scope: os.urandom(32) for scope in scopes}
    header = {"database": database, "epoch": epoch}
    metadata, content_blobs, shares = [], {}, [[], []]
    seen = set()
    for r in sorted(records, key=lambda x:x["id"]):
        require(r["id"] not in seen and r["modality"] in MODALITIES, "Bad or duplicate record")
        seen.add(r["id"])
        encoded = validate_feature(r["embedding"])
        f0, f1 = split_values(encoded)
        shares[0].append(f0); shares[1].append(f1)
        m = {k:r[k] for k in ("id", "scope", "modality")}
        m["feature"] = encrypt_content(keys[r["scope"]],
                encode(r["embedding"].astype("<f4", copy=False).tobytes()), aad(header,m,"feature"))
        content = encrypt_content(keys[r["scope"]], r["content"], aad(header,m,"content"))
        m["content_digest"] = digest(content)
        metadata.append(m); content_blobs[m["id"]] = content
    partitions = [{"scope":s, "modality":m, "count":len(part), "root":merkle_root(part)}
                  for s in scopes for m in sorted(MODALITIES)
                  for part in [[r for r in metadata if r["scope"] == s and r["modality"] == m]]]
    sketch_blobs, catalog = {}, None
    sketch_seconds = 0.0
    if compact:
        from .compact_verify import build_private_sketches, make_catalog
        sketch_started = perf_counter()
        catalog = [make_catalog(m) for m in metadata]
        sketch_blobs = build_private_sketches(records, catalog, partitions, keys, header)
        sketch_seconds = perf_counter()-sketch_started
    snapshot = sign(owner, {**header, "schema":SNAPSHOT_SCHEMA,
        "dimensions":DIM, "ring_bits":BITS, "feature_scale_exponent":149,
        "score_scale_exponent":298, "ranking":"exact score desc, ID asc",
        "publication_policy":POLICY, "public_feature_profile":dict(FEATURE_PROFILE),
        "partitions":partitions,
        **({"compact_verification":"private-linear-sketch.v1",
            "private_sketch_plaintext_bytes":SKETCH_PADDED_BYTES} if compact else {})}, "float-snapshot")
    clients = {}
    for subject, allowed in grants.items():
        allowed = sorted(set(allowed))
        require(set(allowed).issubset(scopes), "Unknown permission scope")
        client_key = new_owner_key()
        credential = sign(owner, {"schema":"ppcmr.float-mpc.credential.v1", "subject":subject,
            "snapshot_digest":digest(snapshot), "allowed_scopes":allowed,
            "query_public":export_public_key(client_key.public_key()),
            "expires_at":int(time.time())+3600}, "float-credential")
        clients[subject] = FloatClient(owner_public, snapshot, credential,
                                {s:keys[s] for s in allowed}, client_key,
                                {key:value for key,value in sketch_blobs.items() if key[0] in allowed})
    server_data = {"owner_public":owner_public, "snapshot":snapshot,
                   "records":metadata, "contents":content_blobs}
    if compact:
        server_data["catalog"] = catalog
        server_data["publication_metrics"] = {"private_sketch_generation_seconds":sketch_seconds}
    return server_data, tuple(shares), clients


def make_query_context(client, modality, k=10):
    require(modality in MODALITIES and type(k) is int and k > 0, "Invalid query or k")
    credential = client.credential["payload"]
    return sign(client.signing_key, {"schema":"ppcmr.float-mpc.query.v2",
        "subject":credential["subject"], "credential_digest":digest(client.credential),
        "snapshot_digest":digest(client.snapshot), "query_modality":modality,
        "target_modality":"text" if modality == "image" else "image",
        "k":k, "nonce":os.urandom(16).hex()}, "float-query")


def make_query(client, feature, modality, k=10):
    encoded = validate_feature(feature)
    qshares = split_values(encoded)
    request = make_query_context(client, modality, k)
    # Each private channel carries only its recipient's binding. Publishing
    # the OTHER share's digest enables offline candidate-query confirmation.
    identifier = digest(request)
    envelopes = tuple(sign(client.signing_key, {
        "schema":"ppcmr.float-mpc.query-share.v2", "request_digest":identifier,
        "party":party, "share_digest":share_digest(qshares[party])},
        "float-query-share") for party in range(2))
    return QuerySession(request, feature.copy(), qshares, envelopes)


def authenticate(owner_public, snapshot, credential, request):
    owner_public = load_public_key(owner_public)
    snap = verify_signature(owner_public, snapshot, "float-snapshot")
    validate_publication(snap)
    require(snap["dimensions"] == DIM and snap["ring_bits"] == BITS, "Unsupported arithmetic")
    cred = verify_signature(owner_public, credential, "float-credential")
    require(cred["snapshot_digest"] == digest(snapshot), "Stale credential/snapshot")
    require(type(cred["expires_at"]) is int and cred["expires_at"] > time.time(), "Expired credential")
    req = verify_signature(load_public_key(cred["query_public"]), request, "float-query")
    require(set(req) == {"schema", "subject", "credential_digest", "snapshot_digest",
        "query_modality", "target_modality", "k", "nonce"}, "Malformed query")
    require(req["schema"] == "ppcmr.float-mpc.query.v2" and
            req["subject"] == cred["subject"] and
            req["credential_digest"] == digest(credential) and
            req["snapshot_digest"] == digest(snapshot), "Wrong request identity/version")
    require(req["query_modality"] in MODALITIES and req["target_modality"] in MODALITIES and
            req["query_modality"] != req["target_modality"], "Wrong target modality")
    require(type(req["k"]) is int and req["k"] > 0 and isinstance(req["nonce"],str) and
            len(req["nonce"]) == 32 and all(c in "0123456789abcdef" for c in req["nonce"]),
            "Invalid request parameters")
    require(cred["allowed_scopes"] == sorted(set(cred["allowed_scopes"])) and
            set(cred["allowed_scopes"]).issubset({p["scope"] for p in snap["partitions"]}), "Invalid rights")
    return snap, cred, req


def authenticate_query_share(credential_payload, request, envelope, party, share):
    """Validate one recipient-private share; never disclose this to the peer."""
    require(type(party) is int and party in (0, 1), "Invalid share recipient")
    payload = verify_signature(load_public_key(credential_payload["query_public"]),
                               envelope, "float-query-share")
    require(set(payload) == {"schema", "request_digest", "party", "share_digest"} and
            payload["schema"] == "ppcmr.float-mpc.query-share.v2", "Malformed query share envelope")
    require(type(payload["party"]) is int and payload["party"] == party, "Wrong share recipient")
    require(payload["request_digest"] == digest(request), "Query share request mismatch")
    require(len(share) == DIM and share_digest(share) == payload["share_digest"],
            "Query share commitment mismatch")
    return payload


def prepare_message(client, session, party):
    require(type(party) is int and party in (0, 1), "Invalid share recipient")
    return {"command":"prepare", "request":session.request,
            "credential":client.credential, "share":session.shares[party],
            "share_envelope":session.share_envelopes[party]}


def validate_witness(client, session, witnesses):
    snap, cred, req = authenticate(client.owner_public, client.snapshot, client.credential, session.request)
    require(isinstance(witnesses,list), "Malformed candidate witness")
    seen = set()
    for m in witnesses:
        require(isinstance(m,dict) and set(m) == FIELDS, "Malformed record")
        require(m["id"] not in seen, "Duplicate candidate")
        seen.add(m["id"])
        require(m["scope"] in cred["allowed_scopes"] and m["modality"] == req["target_modality"],
                "Unauthorized/wrong-modality candidate")
    require([m["id"] for m in witnesses] == sorted(seen), "Noncanonical candidate order")
    for partition in snap["partitions"]:
        if partition["scope"] in cred["allowed_scopes"] and partition["modality"] == req["target_modality"]:
            part = [m for m in witnesses if m["scope"] == partition["scope"]]
            require(len(part) == partition["count"] and merkle_root(part) == partition["root"],
                    "Incomplete or altered authorized partition")
    return snap, cred, req


def expected_scores(client, session, witnesses):
    snap, cred, _ = validate_witness(client,session,witnesses)
    query = validate_feature(session.feature)
    # The trusted client retains both bindings; a server receives only its own.
    require(len(session.shares) == 2 and len(session.share_envelopes) == 2,
            "Malformed client query session")
    for party in range(2):
        authenticate_query_share(cred, session.request, session.share_envelopes[party],
                                 party, session.shares[party])
    require(tuple((x+y)&MASK for x,y in zip(*session.shares)) == tuple(x&MASK for x in query),
            "Client query differs from signed shares")
    result = []
    for m in witnesses:
        data = decode(decrypt_content(client.scope_keys[m["scope"]], m["feature"], aad(snap,m,"feature")))
        require(len(data) == DIM*4, "Malformed encrypted embedding")
        f = np.frombuffer(data,dtype="<f4").copy()
        result.append(exact_dot_encoded(query, validate_feature(f)))
    return result


def verify_result(client, session, result, check_contents=True):
    """Validate the entire candidate set and exact scores, then returned top-k."""
    try:
        require(set(result) == {"request_digest","snapshot_digest","witnesses","scores","results"},
                "Malformed response")
        require(result["request_digest"] == digest(session.request) and
                result["snapshot_digest"] == digest(client.snapshot), "Response request/version mismatch")
        scores = expected_scores(client,session,result["witnesses"])
        require(isinstance(result["scores"],list) and len(scores) == len(result["scores"]) and
                all(type(x) is int for x in result["scores"]) and scores == result["scores"],
                "Missing/incorrect secure scores")
        witnesses = result["witnesses"]
        k = min(session.request["payload"]["k"],len(witnesses))
        order = sorted(range(len(witnesses)), key=lambda i:(-scores[i],witnesses[i]["id"]))[:k]
        selected = [witnesses[i] for i in order]
        require(isinstance(result["results"],list) and len(result["results"]) == k,
                "Missing top-k result")
        require([r["id"] for r in result["results"]] == [m["id"] for m in selected], "Incorrect top-k order/membership")
        contents = []
        for r,m in zip(result["results"],selected):
            require(set(r) == {"id","content"} and digest(r["content"]) == m["content_digest"], "Altered content")
            if check_contents:
                contents.append(decrypt_content(client.scope_keys[m["scope"]],r["content"],
                        aad(client.snapshot["payload"],m,"content")))
        return {"accepted":True,"reason":"OK","candidate_count":len(scores),
                "ids":[m["id"] for m in selected],"contents":contents}
    except Exception as error:
        return {"accepted":False,"reason":str(error),"candidate_count":None,"ids":[],"contents":[]}


def _send(connection, message):
    # Trusted local harness transport. Not an untrusted network serialization.
    data = pickle.dumps(message, protocol=5)
    connection.send_bytes(data)
    return len(data)


def _recv(connection):
    return pickle.loads(connection.recv_bytes())


def _recv_counted(connection):
    data = connection.recv_bytes()
    return pickle.loads(data), len(data)


def _opening_send(connection, binding, values):
    header = canonical(binding)
    message = struct.pack("<I",len(header)) + header + pack_words(values)
    connection.send_bytes(message)
    return len(message)


def _opening_recv(connection, binding, count):
    message = connection.recv_bytes()
    require(len(message) >= 4, "Truncated MPC message")
    hlen = struct.unpack("<I",message[:4])[0]
    require(json.loads(message[4:4+hlen]) == binding, "MPC session/triple binding mismatch")
    body = message[4+hlen:]
    require(len(body) == count*WORD_BYTES, "MPC opening dimension mismatch")
    return [int.from_bytes(body[i:i+WORD_BYTES],"little") for i in range(0,len(body),WORD_BYTES)]


def unpack_words(blob, count):
    require(isinstance(blob,bytes) and len(blob)==count*WORD_BYTES,
            "Packed ring dimension mismatch")
    return [int.from_bytes(blob[i:i+WORD_BYTES],"little")
            for i in range(0,len(blob),WORD_BYTES)]


def _worker(party, control, peer, server_data, database_shares):
    used_nonces, used_triples, prepared, sessions = set(), set(), {}, {}
    cached = None
    while True:
        message = _recv(control)
        command = message.get("command")
        if command == "stop":
            break
        try:
            if command == "cache_matrix":
                require(cached is None,"Matrix cache already initialized; new snapshot requires new workers")
                context=message["context"]
                n=len(server_data["records"])
                require(set(context)=={"stage","mask","snapshot","record_order","rows","dimensions","ring_bits"}
                        and context["stage"]=="immutable-matrix-open.v1"
                        and context["snapshot"]==digest(server_data["snapshot"])
                        and context["record_order"]==digest([m["id"] for m in server_data["records"]])
                        and context["rows"]==n and context["dimensions"]==DIM and context["ring_bits"]==BITS,
                        "Matrix cache version/order mismatch")
                mask=unpack_words(message["A_blob"],n*DIM)
                # Flatten without copying the original per-row feature shares.
                own=[(x-mask[i*DIM+j])&MASK for i,row in enumerate(database_shares) for j,x in enumerate(row)]
                started=perf_counter()
                if party==0:
                    sent=_opening_send(peer,context,own)
                    other=_opening_recv(peer,context,len(own))
                else:
                    other=_opening_recv(peer,context,len(own))
                    sent=_opening_send(peer,context,own)
                opened=[(x+y)&MASK for x,y in zip(own,other)]
                matrix=[opened[i:i+DIM] for i in range(0,len(opened),DIM)]
                from .exact_ring import matrix_digits
                cached={"context":context,"D":matrix,"D_digits":matrix_digits(matrix),
                        "X_digits":matrix_digits(database_shares),"selected":None}
                # Original X_t shares need not equal A_t+1_{t=0}D.
                # Summation gives c+D*b+X*E=A*b+D*b+X*(q-b)=X*q.
                _send(control,{"ok":True,"peer_sent_bytes":sent,
                    "opened_digest":hashlib.sha256(pack_words(opened)).hexdigest(),
                    "seconds":perf_counter()-started,"pid":os.getpid()})
                del mask,own,other,opened
            elif command == "prepare":
                snap, cred, req = authenticate(server_data["owner_public"],server_data["snapshot"],
                                               message["credential"],message["request"])
                require(req["nonce"] not in used_nonces, "Replayed query")
                authenticate_query_share(cred, message["request"], message["share_envelope"],
                                         party, message["share"])
                used_nonces.add(req["nonce"])
                indices = [i for i,m in enumerate(server_data["records"])
                    if m["scope"] in cred["allowed_scopes"] and m["modality"] == req["target_modality"]]
                mode = message.get("witness_mode","full")
                require(mode in {"full","compact"}, "Invalid witness mode")
                require(mode != "compact" or "catalog" in server_data, "Compact verification not published")
                witness_records = server_data["catalog"] if mode == "compact" else server_data["records"]
                identifier = digest(message["request"])
                prepared[identifier] = (indices,message["share"],req)
                sessions[identifier] = {"indices":indices,"credential":message["credential"]}
                _send(control,{"ok":True,"ids":[server_data["records"][i]["id"] for i in indices],
                    "witnesses":[witness_records[i] for i in indices] if party == 0 else None})
            elif command == "abort":
                prepared.pop(message["request_digest"],None)
                _send(control,{"ok":True})
            elif command == "compute":
                identifier = message["request_digest"]
                indices, query_share, req = prepared.pop(identifier)
                triple = message["triple"]
                require(triple["id"] not in used_triples, "Reused preprocessing triple")
                used_triples.add(triple["id"])
                n = len(indices)
                require(len(triple["A"]) == n and len(triple["c"]) == n, "Triple row count mismatch")
                started = perf_counter()
                peer_bytes = 0
                if n:
                    require(len(triple["b"]) == DIM and all(len(r)==DIM for r in triple["A"]), "Triple dimension mismatch")
                    xrows = [database_shares[i] for i in indices]
                    d_share = [(x-a)&MASK for row,a_row in zip(xrows,triple["A"]) for x,a in zip(row,a_row)]
                    e_share = [(q-b)&MASK for q,b in zip(query_share,triple["b"])]
                    opening = d_share + e_share
                    binding = {"request":identifier,"triple":triple["id"],"rows":n,"dimensions":DIM}
                    if party == 0:
                        peer_bytes = _opening_send(peer,binding,opening)
                        other = _opening_recv(peer,binding,len(opening))
                    else:
                        other = _opening_recv(peer,binding,len(opening))
                        peer_bytes = _opening_send(peer,binding,opening)
                    opened = [(a+b)&MASK for a,b in zip(opening,other)]
                    e = opened[n*DIM:]
                    values = []
                    for j in range(n):
                        d = opened[j*DIM:(j+1)*DIM]
                        value = triple["c"][j] + sum(v*b+a*w for v,b,a,w in
                                    zip(d,triple["b"],triple["A"][j],e))
                        if party == 0:
                            value += sum(v*w for v,w in zip(d,e))
                        values.append(value&MASK)
                else:
                    values = []
                _send(control,{"ok":True,"request_digest":identifier,"shares":values,
                    "seconds":perf_counter()-started,"peer_sent_bytes":peer_bytes,"pid":os.getpid()})
            elif command == "compute_cached":
                require(cached is not None,"Matrix cache not initialized")
                identifier=message["request_digest"]
                indices,query_share,req=prepared.pop(identifier)
                triple=message["triple"]
                require(set(triple)=={"id","mask","snapshot","b_blob","c_blob"}
                        and triple["mask"]==cached["context"]["mask"]
                        and triple["snapshot"]==cached["context"]["snapshot"],"Wrong matrix-cache preprocessing context")
                require(triple["id"] not in used_triples,"Reused preprocessing triple")
                used_triples.add(triple["id"])
                n=len(indices)
                b=unpack_words(triple["b_blob"],DIM if n else 0)
                c=unpack_words(triple["c_blob"],n)
                kernel=message["kernel"]
                require(kernel in {"python","limb-fp64"},"Unsupported exact ring kernel")
                started=perf_counter(); peer_bytes=0
                if n:
                    own=[(q-v)&MASK for q,v in zip(query_share,b)]
                    binding={"stage":"cached-query-open.v1","request":identifier,"triple":triple["id"],
                        "mask":triple["mask"],"snapshot":triple["snapshot"],"rows":n,"dimensions":DIM,
                        "candidates":digest(indices),"kernel":kernel}
                    if party==0:
                        peer_bytes=_opening_send(peer,binding,own)
                        other=_opening_recv(peer,binding,DIM)
                    else:
                        other=_opening_recv(peer,binding,DIM)
                        peer_bytes=_opening_send(peer,binding,own)
                    e=[(x+y)&MASK for x,y in zip(own,other)]
                    # X_t denotes the ORIGINAL database shares, not A_t.
                    # c=A*b stays input-independent; D*b is required.
                    if kernel=="python":
                        values=[(c[j]+sum(v*w+x*z for v,w,x,z in
                                zip(cached["D"][i],b,database_shares[i],e)))&MASK
                                for j,i in enumerate(indices)]
                    else:
                        from .exact_ring import matvec
                        key=tuple(indices)
                        if cached["selected"] is None or cached["selected"][0]!=key:
                            cached["selected"]=(key,cached["X_digits"][indices],cached["D_digits"][indices])
                        products=matvec(cached["selected"][1],e)
                        masked=matvec(cached["selected"][2],b)
                        values=[(x+y+z)&MASK for x,y,z in zip(c,products,masked)]
                else:
                    values=[]
                _send(control,{"ok":True,"request_digest":identifier,"shares":values,
                    "seconds":perf_counter()-started,"peer_sent_bytes":peer_bytes,"pid":os.getpid()})
            elif command == "fetch":
                stored = sessions[message["request_digest"]]
                cred = verify_signature(load_public_key(server_data["owner_public"]),stored["credential"],"float-credential")
                require(cred["expires_at"] > time.time(), "Expired credential")
                payload = verify_signature(load_public_key(cred["query_public"]),message["fetch"],"float-fetch")
                require(payload["request_digest"] == message["request_digest"], "Wrong fetch request")
                ids = payload["ids"]
                allowed = {server_data["records"][i]["id"] for i in stored["indices"]}
                require(isinstance(ids,list) and len(set(ids)) == len(ids) and set(ids).issubset(allowed), "Unauthorized content fetch")
                _send(control,{"ok":True,"results":[{"id":i,"content":server_data["contents"][i]} for i in ids]})
            else:
                raise ValueError("Unknown cloud command")
        except Exception as error:
            _send(control,{"ok":False,"error":str(error)})
    control.close(); peer.close()


class TwoCloudEngine:
    compute_command="compute"
    def __init__(self, server_data, shares):
        snapshot = verify_signature(load_public_key(server_data['owner_public']),
                                    server_data['snapshot'], 'float-snapshot')
        validate_publication(snapshot)
        self.server_data = server_data
        ctx = mp.get_context("spawn")
        peer0, peer1 = ctx.Pipe(duplex=True)
        self.connections, self.processes = [], []
        started = perf_counter()
        for i,peer in enumerate((peer0,peer1)):
            parent,child = ctx.Pipe(duplex=True)
            process = ctx.Process(target=_worker,args=(i,child,peer,server_data,shares[i]))
            process.start(); child.close()
            self.connections.append(parent); self.processes.append(process)
        peer0.close(); peer1.close()
        self.startup_seconds = perf_counter()-started

    def close(self):
        for process,connection in zip(self.processes,self.connections):
            if process.is_alive():
                _send(connection,{"command":"stop"})
        for process,connection in zip(self.processes,self.connections):
            process.join(timeout=10)
            if process.is_alive():
                process.terminate(); process.join(timeout=5)
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()

    def _responses(self):
        responses = []
        for connection in self.connections:
            require(connection.poll(60), "Cloud response timeout")
            response, size = _recv_counted(connection)
            self.received_bytes += size
            responses.append(response)
        return responses

    def _make_material(self,witnesses):
        return make_triple(len(witnesses))

    def _compute_message(self,identifier,material):
        return {"command":self.compute_command,"request_digest":identifier,"triple":material}

    def search(self, client, session, witness_mode="full"):
        require(witness_mode in {"full","compact"}, "Invalid witness mode")
        self.received_bytes = 0
        identifier = digest(session.request)
        started = perf_counter()
        client_bytes = 0
        for i,connection in enumerate(self.connections):
            message = prepare_message(client,session,i)
            message["witness_mode"] = witness_mode
            client_bytes += _send(connection,message)
        prepared = self._responses()
        if not all(r["ok"] for r in prepared):
            for connection in self.connections:
                _send(connection,{"command":"abort","request_digest":identifier})
            self._responses()
            raise ValueError("; ".join(r.get("error","") for r in prepared if not r["ok"]))
        witnesses = prepared[0]["witnesses"]
        try:
            if witness_mode == "compact":
                from .compact_verify import validate_catalog
                validate_catalog(client,session,witnesses)
            else:
                validate_witness(client,session,witnesses)
            ids = [m["id"] for m in witnesses]
            require(all(r["ids"] == ids for r in prepared), "Cloud candidate sets disagree")
        except Exception:
            for connection in self.connections:
                _send(connection,{"command":"abort","request_digest":identifier})
            self._responses()
            raise
        prepare_seconds = perf_counter()-started
        offline_start = perf_counter()
        triples = self._make_material(witnesses)
        offline_seconds = perf_counter()-offline_start
        online_start = perf_counter()
        offline_bytes = 0
        for i,connection in enumerate(self.connections):
            offline_bytes += _send(connection,self._compute_message(identifier,triples[i]))
        computed = self._responses()
        require(all(r["ok"] and r["request_digest"] == identifier for r in computed), "Cloud computation failed")
        require(all(len(r["shares"]) == len(witnesses) for r in computed), "Incomplete score shares")
        scores = []
        for a,b in zip(computed[0]["shares"],computed[1]["shares"]):
            require(type(a) is int and type(b) is int and 0<=a<=MASK and 0<=b<=MASK, "Invalid output shares")
            value = (a+b)&MASK
            scores.append(value-MODULUS if value >= HALF else value)
        k = min(session.request["payload"]["k"],len(ids))
        order = sorted(range(len(ids)),key=lambda i:(-scores[i],ids[i]))[:k]
        selected = [ids[i] for i in order]
        online_seconds = perf_counter()-online_start
        fetch = sign(client.signing_key,{"request_digest":identifier,"ids":selected},"float-fetch")
        client_bytes += _send(self.connections[0],{"command":"fetch","request_digest":identifier,"fetch":fetch})
        require(self.connections[0].poll(60), "Content fetch timeout")
        fetched, size = _recv_counted(self.connections[0])
        self.received_bytes += size
        require(fetched["ok"], "Content fetch denied")
        result = {"request_digest":identifier,"snapshot_digest":digest(client.snapshot),
                  "witnesses":witnesses,"scores":scores,"results":fetched["results"]}
        if witness_mode == "compact":
            result["proof_format"] = "private-linear-sketch.v1"
        return result, {"prepare_seconds":prepare_seconds,"trusted_preprocessing_seconds":offline_seconds,
            "online_seconds_including_triple_delivery":online_seconds,
            "end_to_end_seconds_without_verification":perf_counter()-started,
            "client_request_sent_bytes":client_bytes,"trusted_preprocessing_sent_bytes":offline_bytes,
            "client_response_received_bytes":self.received_bytes,
            "client_total_transport_payload_bytes":client_bytes+self.received_bytes,
            "peer_opening_sent_bytes":sum(r["peer_sent_bytes"] for r in computed),
            "server_cpu_phase_seconds":[r["seconds"] for r in computed],
            "cloud_process_ids":[r["pid"] for r in computed],
            "result_json_bytes":len(canonical(result)),
            "candidate_count":len(ids),"topk_location":"client"}
