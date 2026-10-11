"""Owner-preprocessed private linear checks over a prime field.

Research candidate, not a public proof or novel cryptographic primitive.
Checks exact FP32-value scores without downloading each feature. Security
requires an honest owner, private verifier seeds/columns, HMAC-SHA512 PRF
security, authentic state, and no verifier/cloud collusion. Full client-side
score vectors and O(n log n) ranking remain. The score-error bound in the ideal
PRF game is at most 2^-512 per attempted invalid response; concrete security
also includes primitive advantages and adaptive-attempt accounting.
"""
import hashlib
import hmac
import json
import os
from time import perf_counter

from float_reference import encode_fp32
from .crypto import (canonical, decode, decrypt_content, digest, encode,
                     encrypt_content, merkle_root)
from .float_mpc import (DIM, MASK, aad, authenticate, authenticate_query_share,
                        require, validate_feature)
from .publication_policy import SKETCH_PADDED_BYTES, validate_publication

PRIME = (1 << 521) - 1
SCORE_BOUND = 1 << 307
FORMAT = 'private-linear-sketch.v1'
CATALOG_FIELDS = {'id','scope','modality','feature_digest','content_digest'}


def make_catalog(metadata):
    return {k:metadata[k] for k in ('id','scope','modality','content_digest')} | {
        'feature_digest':digest(metadata['feature'])}


def sketch_context(snapshot, scope, modality):
    return {'database':snapshot['database'],'epoch':snapshot['epoch'],
            'scope':scope,'modality':modality,'purpose':'private-linear-sketch.v1'}


def coefficient(seed, context, identifier):
    # Integers in [0,2^512), injected into F_p with p>2^512.
    message = canonical({**context,'id':identifier,'purpose':'score-check-coefficient.v1'})
    return int.from_bytes(hmac.new(seed,message,hashlib.sha512).digest(),'big')


def build_private_sketches(records, catalog, partitions, keys, header):
    by_id = {r['id']:r for r in records}
    blobs = {}
    for part in partitions:
        scope,modality = part['scope'],part['modality']
        members = [m for m in catalog if m['scope']==scope and m['modality']==modality]
        root = merkle_root(members)
        seed = os.urandom(32)
        context = sketch_context(header,scope,modality)
        columns = [0]*DIM
        for m in members:
            weight = coefficient(seed,context,m['id'])
            values = encode_fp32(by_id[m['id']]['embedding'])
            columns = [(a+weight*x)%PRIME for a,x in zip(columns,values)]
        plaintext = {'schema':FORMAT,'catalog_root':root,'count':len(members),
                     'field_bits':521,'seed':encode(seed),'columns':columns}
        serialized = json.dumps(plaintext,separators=(',',':'))
        require(len(serialized.encode('utf-8')) <= SKETCH_PADDED_BYTES,
                'Private sketch exceeds public padding budget')
        # All fields are ASCII. JSON permits trailing whitespace, so this
        # pads the encrypted plaintext without changing its parsed content.
        serialized += ' ' * (SKETCH_PADDED_BYTES-len(serialized))
        blob = encrypt_content(keys[scope],serialized,context)
        part['catalog_root'] = root
        part['sketch_digest'] = digest(blob)
        blobs[(scope,modality)] = blob
    return blobs


def validate_catalog(client, session, witnesses, check_coverage=True):
    snap,cred,req = authenticate(client.owner_public,client.snapshot,client.credential,session.request)
    require(snap.get('compact_verification')==FORMAT,'Compact verification not published')
    require(isinstance(witnesses,list),'Malformed catalog')
    seen = set()
    for m in witnesses:
        require(isinstance(m,dict) and set(m)==CATALOG_FIELDS,'Malformed catalog record')
        require(isinstance(m['id'],str) and m['id'] not in seen,'Duplicate or invalid candidate')
        seen.add(m['id'])
        require(m['scope'] in cred['allowed_scopes'] and m['modality']==req['target_modality'],
                'Unauthorized/wrong-modality candidate')
        for field in ('feature_digest','content_digest'):
            require(isinstance(m[field],str) and len(m[field])==64 and
                    all(c in '0123456789abcdef' for c in m[field]),'Malformed catalog digest')
    require([m['id'] for m in witnesses]==sorted(seen),'Noncanonical candidate order')
    if check_coverage:
        for part in snap['partitions']:
            if part['scope'] in cred['allowed_scopes'] and part['modality']==req['target_modality']:
                members = [m for m in witnesses if m['scope']==part['scope']]
                require(len(members)==part['count'] and merkle_root(members)==part['catalog_root'],
                        'Incomplete or altered authorized catalog')
    return snap,cred,req


def prepare_verifier_state(client):
    """Authenticate/decrypt epoch-specific state once, before measured queries.

    Returns a cache bound to the exact signed snapshot. The caller must keep
    the seed and columns private, just as scope decryption keys are private.
    No features or per-candidate coefficients are retained in this cache.
    """
    from .crypto import load_public_key, verify_signature
    snap = verify_signature(load_public_key(client.owner_public),client.snapshot,'float-snapshot')
    validate_publication(snap)
    cred = verify_signature(load_public_key(client.owner_public),client.credential,'float-credential')
    require(cred['snapshot_digest']==digest(client.snapshot),'Stale verifier state')
    require(snap.get('compact_verification')==FORMAT,'Compact verification not published')
    state = {'snapshot_digest':digest(client.snapshot),'partitions':{}}
    for part in snap['partitions']:
        if part['scope'] not in cred['allowed_scopes']:
            continue
        key = (part['scope'],part['modality'])
        blob = client.verification_sketches[key]
        require(digest(blob)==part['sketch_digest'],'Wrong private verifier ciphertext')
        context = sketch_context(snap,*key)
        plaintext = decrypt_content(client.scope_keys[part['scope']],blob,context)
        require(len(plaintext.encode('utf-8')) == SKETCH_PADDED_BYTES,
                'Wrong private sketch padding length')
        data = json.loads(plaintext)
        require(set(data)=={'schema','catalog_root','count','field_bits','seed','columns'} and
                data['schema']==FORMAT and data['catalog_root']==part['catalog_root'] and
                data['count']==part['count'] and data['field_bits']==521,'Wrong private verifier context')
        seed = decode(data['seed'])
        require(len(seed)==32 and isinstance(data['columns'],list) and len(data['columns'])==DIM and
                all(type(x) is int and 0<=x<PRIME for x in data['columns']),'Malformed private verifier state')
        state['partitions'][key] = {'seed':seed,'columns':data['columns'],'context':context}
    return state


def verify_compact(client, session, result, state=None, *, check_coverage=True,
                   check_scores=True, check_ranking=True):
    """Flags are solely for controlled research ablations; defaults are full.

    Do not expose flags at an application endpoint. Every accepted production
    result requires all checks. No residual, seed, or column is returned.
    """
    started = perf_counter()
    try:
        require(set(result)=={'request_digest','snapshot_digest','witnesses','scores','results','proof_format'} and
                result['proof_format']==FORMAT,'Malformed compact response')
        require(result['request_digest']==digest(session.request) and
                result['snapshot_digest']==digest(client.snapshot),'Response request/version mismatch')
        snap,cred,req = validate_catalog(client,session,result['witnesses'],check_coverage)
        query = validate_feature(session.feature)
        require(len(session.shares)==2 and len(session.share_envelopes)==2,'Malformed client query session')
        for party in (0,1):
            authenticate_query_share(cred,session.request,session.share_envelopes[party],party,session.shares[party])
        require(tuple((a+b)&MASK for a,b in zip(*session.shares))==tuple(x&MASK for x in query),
                'Client query differs from signed shares')
        witnesses,scores = result['witnesses'],result['scores']
        require(isinstance(scores,list) and len(scores)==len(witnesses) and
                all(type(s) is int and -SCORE_BOUND<=s<=SCORE_BOUND for s in scores),
                'Missing/out-of-range exact score')
        if check_scores:
            state = prepare_verifier_state(client) if state is None else state
            require(state['snapshot_digest']==digest(client.snapshot),'Stale verifier cache')
            for part in snap['partitions']:
                key = (part['scope'],part['modality'])
                if part['scope'] not in cred['allowed_scopes'] or part['modality']!=req['target_modality']:
                    continue
                private = state['partitions'][key]
                left = sum(coefficient(private['seed'],private['context'],m['id'])*s
                           for m,s in zip(witnesses,scores) if m['scope']==part['scope'])%PRIME
                right = sum(c*q for c,q in zip(private['columns'],query))%PRIME
                require(left==right,'Invalid compact score relation')
        count = min(req['k'],len(witnesses))
        require(isinstance(result['results'],list) and len(result['results'])==count,'Missing top-k result')
        order = sorted(range(len(witnesses)),key=lambda i:(-scores[i],witnesses[i]['id']))[:count]
        if check_ranking:
            require([r['id'] for r in result['results']]==[witnesses[i]['id'] for i in order],
                    'Incorrect top-k order/membership')
        lookup = {m['id']:m for m in witnesses}
        require(len({r['id'] for r in result['results']})==count,'Duplicate result')
        contents = []
        for r in result['results']:
            require(set(r)=={'id','content'} and r['id'] in lookup,'Invalid result membership')
            m = lookup[r['id']]
            require(digest(r['content'])==m['content_digest'],'Altered content')
            contents.append(decrypt_content(client.scope_keys[m['scope']],r['content'],aad(snap,m,'content')))
        return {'accepted':True,'reason':'OK','ids':[r['id'] for r in result['results']],
                'contents':contents,'candidate_count':len(witnesses),'seconds':perf_counter()-started}
    except Exception as error:
        return {'accepted':False,'reason':str(error),'ids':[],'contents':[],
                'candidate_count':None,'seconds':perf_counter()-started}
