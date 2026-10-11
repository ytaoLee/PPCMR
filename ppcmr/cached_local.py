"""Authenticated, epoch-bound client feature caching for exact local retrieval.

No query feature goes to the worker. The initial complete encrypted collection
is authenticated and decrypted once. Each later request is authenticated and
bound to its eligible ordered collection before the client scores cached values.
Python and bounded signed-limb FP64 kernels compute the same exact integers.
"""
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from float_reference import exact_dot_encoded
from .crypto import canonical,decode,decrypt_content,digest,load_public_key,sign,verify_signature
from .float_mpc import (DIM,HALF,MASK,MODULUS,_recv,_send,aad,authenticate,
                        require,validate_feature,validate_witness)
from .local_reference import AuthorizedLocalEngine,make_local_query

LOCAL_LIMBS=10
BASE=1<<16


def signed_limb_matrix(rows):
    """Signed-magnitude digits; encoded |FP32|<=1 needs at most 150 bits."""
    require(all(len(row)==DIM for row in rows),'Expected 512-column local matrix')
    require(all(type(v) is int and abs(v)<=1<<149 for row in rows for v in row),
            'Invalid exact local feature value')
    blob=b''.join(abs(v).to_bytes(LOCAL_LIMBS*2,'little') for row in rows for v in row)
    result=np.frombuffer(blob,dtype='<u2').reshape(len(rows),DIM,LOCAL_LIMBS).astype(np.int32)
    signs=np.fromiter((-1 if v<0 else 1 for row in rows for v in row),dtype=np.int32,
                      count=len(rows)*DIM).reshape(len(rows),DIM,1)
    result*=signs
    return result


def exact_local_matvec(matrix,vector):
    require(isinstance(matrix,np.ndarray) and matrix.dtype==np.int32 and
            matrix.ndim==3 and matrix.shape[1:]==(DIM,LOCAL_LIMBS),
            'Invalid cached local limb format')
    require(not matrix.size or (matrix.min()>-BASE and matrix.max()<BASE),
            'Local limb out of range')
    right=signed_limb_matrix([vector])[0].astype(np.float64)
    n=len(matrix)
    if not n:return []
    coefficients=np.zeros((n,20),dtype=np.int64)
    # Every signed limb dot product and partial sum has magnitude <2^41.
    # At most ten contributions per coefficient remain below 2^45.
    for i in range(LOCAL_LIMBS):
        products=np.ascontiguousarray(matrix[:,:,i],dtype=np.float64) @ right
        require(products.dtype==np.float64,'Genuine FP64 BLAS required')
        coefficients[:,i:i+LOCAL_LIMBS]+=products.astype(np.int64)
    carry=np.zeros(n,dtype=np.int64)
    output=np.zeros((n,20),dtype='<u2')
    for i in range(20):
        value=coefficients[:,i]+carry
        output[:,i]=(value & (BASE-1)).astype('<u2')
        carry=value>>16  # Signed floor division also propagates negative values.
    ring=[int.from_bytes(row.tobytes(),'little') for row in output]
    return [v-MODULUS if v>=HALF else v for v in ring]


def _cached_local_worker(control,server_data):
    used_nonces,sessions,collections=set(),{},{}
    while True:
        message=_recv(control)
        command=message.get('command')
        if command=='stop':break
        try:
            if command in {'initialize_cache','prepare_cached'}:
                _,cred,req=authenticate(server_data['owner_public'],server_data['snapshot'],
                                       message['credential'],message['request'])
                require(req['nonce'] not in used_nonces,'Replayed query')
                used_nonces.add(req['nonce'])
                key=(tuple(cred['allowed_scopes']),req['target_modality'])
                if key not in collections:
                    witnesses=[m for m in server_data['records'] if m['scope'] in cred['allowed_scopes']
                               and m['modality']==req['target_modality']]
                    collections[key]=(witnesses,digest(witnesses),{m['id'] for m in witnesses})
                witnesses,collection_digest,allowed=collections[key]
                identifier=digest(message['request'])
                sessions[identifier]=(cred,allowed)
                response={'ok':True,'request_digest':identifier,'snapshot_digest':digest(server_data['snapshot']),
                          'collection_digest':collection_digest,'candidate_count':len(witnesses),'pid':os.getpid()}
                if command=='initialize_cache':response['witnesses']=witnesses
                _send(control,response)
            elif command=='fetch':
                cred,allowed=sessions[message['request_digest']]
                require(cred['expires_at']>time.time(),'Expired credential')
                payload=verify_signature(load_public_key(cred['query_public']),message['fetch'],'float-fetch')
                require(set(payload)=={'request_digest','ids'} and
                        payload['request_digest']==message['request_digest'],'Wrong fetch request')
                ids=payload['ids']
                require(isinstance(ids,list) and len(ids)==len(set(ids)) and set(ids).issubset(allowed),
                        'Unauthorized content fetch')
                _send(control,{'ok':True,'results':[{'id':i,'content':server_data['contents'][i]} for i in ids]})
            else:raise ValueError('Unknown cached-local command')
        except Exception as error:_send(control,{'ok':False,'error':str(error)})
    control.close()


@dataclass
class LocalFeatureState:
    snapshot_digest:str
    credential_digest:str
    target_modality:str
    collection_digest:str
    witnesses:list
    encoded:list
    digits:np.ndarray


class CachedLocalEngine(AuthorizedLocalEngine):
    def __init__(self,server_data,kernel='signed-limb-fp64'):
        require(kernel in {'python','signed-limb-fp64'},'Unsupported local exact kernel')
        self.kernel=kernel
        context=mp.get_context('spawn')
        self.connection,child=context.Pipe(duplex=True)
        self.process=context.Process(target=_cached_local_worker,args=(child,server_data))
        started=perf_counter();self.process.start();child.close()
        self.startup_seconds=perf_counter()-started

    def initialize(self,client,modality):
        started=perf_counter()
        session=make_local_query(client,np.zeros(DIM,dtype=np.float32),modality)
        sent=_send(self.connection,{'command':'initialize_cache','request':session.request,'credential':client.credential})
        response,received=self._receive()
        witnesses=response['witnesses']
        snap,_,req=validate_witness(client,session,witnesses)
        require(response['request_digest']==digest(session.request) and
                response['snapshot_digest']==digest(client.snapshot) and
                response['collection_digest']==digest(witnesses) and
                response['candidate_count']==len(witnesses),'Cache delivery binding mismatch')
        encoded=[]
        for m in witnesses:
            raw=decode(decrypt_content(client.scope_keys[m['scope']],m['feature'],aad(snap,m,'feature')))
            require(len(raw)==DIM*4,'Malformed encrypted embedding')
            encoded.append(validate_feature(np.frombuffer(raw,dtype='<f4').copy()))
        digits=signed_limb_matrix(encoded)
        state=LocalFeatureState(digest(client.snapshot),digest(client.credential),req['target_modality'],
                                digest(witnesses),witnesses,encoded,digits)
        return state,{'initialization_seconds':perf_counter()-started,
            'client_request_sent_bytes':sent,'client_response_received_bytes':received,
            'client_total_transport_payload_bytes':sent+received,'candidate_count':len(witnesses),
            'signed_limb_cache_bytes':digits.nbytes,'encoded_values':len(witnesses)*DIM}

    @staticmethod
    def _validate_binding(client,session,state,response):
        require(response['request_digest']==digest(session.request) and
                response['snapshot_digest']==state.snapshot_digest and
                response['collection_digest']==state.collection_digest and
                response['candidate_count']==len(state.witnesses),'Cached collection binding mismatch')

    def search_cached(self,client,session,state):
        started=perf_counter()
        snap,_,req=authenticate(client.owner_public,client.snapshot,client.credential,session.request)
        require(state.snapshot_digest==digest(client.snapshot) and
                state.credential_digest==digest(client.credential) and
                state.target_modality==req['target_modality'],'Stale/wrong-permission local feature cache')
        query=validate_feature(session.feature)
        sent=_send(self.connection,{'command':'prepare_cached','request':session.request,'credential':client.credential})
        prepared,received=self._receive()
        self._validate_binding(client,session,state,prepared)
        prepared_seconds=perf_counter()-started
        scoring_started=perf_counter()
        require(self.kernel in {'python','signed-limb-fp64'},'Unsupported local exact kernel')
        scores=([exact_dot_encoded(query,row) for row in state.encoded] if self.kernel=='python'
                else exact_local_matvec(state.digits,query))
        order=sorted(range(len(scores)),key=lambda i:(-scores[i],state.witnesses[i]['id']))[:req['k']]
        selected=[state.witnesses[i] for i in order]
        scoring_seconds=perf_counter()-scoring_started
        identifier=digest(session.request)
        fetch=sign(client.signing_key,{'request_digest':identifier,'ids':[m['id'] for m in selected]},'float-fetch')
        fetch_started=perf_counter()
        sent+=_send(self.connection,{'command':'fetch','request_digest':identifier,'fetch':fetch})
        fetched,size=self._receive();received+=size
        require([r['id'] for r in fetched['results']]==[m['id'] for m in selected],'Incorrect fetched results')
        contents=[]
        for r,m in zip(fetched['results'],selected):
            require(set(r)=={'id','content'} and digest(r['content'])==m['content_digest'],'Altered content')
            contents.append(decrypt_content(client.scope_keys[m['scope']],r['content'],aad(snap,m,'content')))
        fetch_seconds=perf_counter()-fetch_started
        result={'request_digest':identifier,'snapshot_digest':state.snapshot_digest,'results':fetched['results']}
        response_json_bytes=len(canonical(result))
        return {'accepted':True,'ids':[m['id'] for m in selected],'scores':scores,'contents':contents},{
            'cached_request_authentication_seconds':prepared_seconds,'exact_scoring_and_ranking_seconds':scoring_seconds,
            'content_fetch_check_seconds':fetch_seconds,'search_seconds_including_serialization':perf_counter()-started,
            'client_request_sent_bytes':sent,'client_response_received_bytes':received,
            'client_total_transport_payload_bytes':sent+received,'result_json_bytes':response_json_bytes,
            'candidate_count':len(scores),'server_process_id':prepared['pid'],'local_kernel':self.kernel,
            'features_downloaded_this_query':False,'feature_cache_per_immutable_snapshot':True}
