"""Authenticated local cache delivered/scored in bounded windows."""
from dataclasses import dataclass
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import time
from time import perf_counter
import numpy as np
from .crypto import canonical,decode,decrypt_content,digest,sign,load_public_key,verify_signature
from .float_mpc import DIM,FIELDS,_send,_recv,_recv_counted,aad,authenticate,require,validate_feature
from .local_reference import AuthorizedLocalEngine,make_local_query
from .cached_local import signed_limb_matrix,exact_local_matvec
from .streaming_store import RingFile,root_from_leaves


def _local_worker(control,server,block_rows):
    used=set();collections={};sessions={};initializing=None
    try:
        while True:
            message=_recv(control);command=message.get('command')
            if command=='stop':break
            try:
                if command in {'initialize','prepare_cached'}:
                    _,cred,req=authenticate(server['owner_public'],server['snapshot'],message['credential'],message['request'])
                    require(req['nonce'] not in used,'Replayed query');used.add(req['nonce'])
                    identifier=digest(message['request']);key=(tuple(cred['allowed_scopes']),req['target_modality'])
                    if command=='initialize':
                        records=[r for r in server['records'] if r['scope'] in cred['allowed_scopes'] and r['modality']==req['target_modality']]
                        initializing={'records':records,'key':key,'cursor':0,'hasher':hashlib.sha256(b'['),'first':True}
                        _send(control,{'ok':True,'request_digest':identifier,'snapshot_digest':digest(server['snapshot']),
                            'candidate_count':len(records),'pid':os.getpid()})
                    else:
                        require(key in collections,'Local cache unavailable')
                        records,collection_digest=collections[key]
                        sessions={identifier:(cred,{r['id'] for r in records})}
                        _send(control,{'ok':True,'request_digest':identifier,'snapshot_digest':digest(server['snapshot']),
                            'collection_digest':collection_digest,'candidate_count':len(records),'pid':os.getpid()})
                elif command=='next_cache_block':
                    require(initializing is not None,'No cache delivery')
                    start=initializing['cursor'];records=initializing['records']
                    if start<len(records):
                        witnesses=[server['metadata_store'][r['id']] for r in records[start:start+block_rows]]
                        for m in witnesses:
                            if not initializing['first']:initializing['hasher'].update(b',')
                            initializing['hasher'].update(canonical(m));initializing['first']=False
                        initializing['cursor']+=len(witnesses)
                        _send(control,{'ok':True,'start':start,'witnesses':witnesses,'complete':False})
                    else:
                        initializing['hasher'].update(b']');collection_digest=initializing['hasher'].hexdigest()
                        collections[initializing['key']]=(records,collection_digest);initializing=None
                        _send(control,{'ok':True,'complete':True,'collection_digest':collection_digest})
                elif command=='fetch':
                    identifier=message['request_digest'];cred,allowed=sessions[identifier]
                    require(cred['expires_at']>time.time(),'Expired credential')
                    payload=verify_signature(load_public_key(cred['query_public']),message['fetch'],'float-fetch')
                    require(set(payload)=={'request_digest','ids'} and payload['request_digest']==identifier,'Wrong fetch request')
                    ids=payload['ids'];require(isinstance(ids,list) and len(ids)==len(set(ids)) and set(ids).issubset(allowed),'Unauthorized fetch')
                    _send(control,{'ok':True,'results':[{'id':i,'content':server['contents'][i]} for i in ids]})
                else:raise ValueError('Unknown streamed-local command')
            except Exception as error:_send(control,{'ok':False,'error':str(error)})
    finally:server['metadata_store'].close();server['contents'].close();control.close()


@dataclass
class StreamLocalState:
    snapshot_digest:str
    credential_digest:str
    target_modality:str
    collection_digest:str
    witnesses:list
    digits:RingFile


class StreamingLocalEngine(AuthorizedLocalEngine):
    def __init__(self,server,folder,block_rows=256):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.block_rows=block_rows
        ctx=mp.get_context('spawn');self.connection,child=ctx.Pipe(duplex=True)
        self.process=ctx.Process(target=_local_worker,args=(child,server,block_rows))
        t=perf_counter();self.process.start();child.close();self.startup_seconds=perf_counter()-t

    def initialize(self,client,modality):
        start=perf_counter();session=make_local_query(client,np.zeros(DIM,dtype=np.float32),modality)
        snap,cred,req=authenticate(client.owner_public,client.snapshot,client.credential,session.request)
        sent=_send(self.connection,{'command':'initialize','request':session.request,'credential':client.credential})
        response,received=self._receive();identifier=digest(session.request)
        require(response['request_digest']==identifier and response['snapshot_digest']==digest(client.snapshot),'Wrong cache delivery context')
        n=response['candidate_count'];digits=RingFile(self.folder/f'{modality}_{digest(client.credential)[:16]}_local.bin',n,create=True,dtype='<i4',limbs=10)
        partitions={p['scope']:p for p in snap['partitions'] if p['scope'] in cred['allowed_scopes'] and p['modality']==req['target_modality']}
        leaves={s:[] for s in partitions};witnesses=[];hasher=hashlib.sha256(b'[');previous=None
        while True:
            sent+=_send(self.connection,{'command':'next_cache_block'});block,size=self._receive();received+=size
            if block['complete']:
                hasher.update(b']');require(block['collection_digest']==hasher.hexdigest(),'Wrong delivered collection digest');break
            require(block['start']==len(witnesses) and 0<len(block['witnesses'])<=self.block_rows,'Wrong cache block order')
            encoded=[]
            for m in block['witnesses']:
                require(set(m)==FIELDS and m['scope'] in partitions and m['modality']==req['target_modality'],'Invalid cache member')
                require(previous is None or previous<m['id'],'Duplicate/noncanonical cache member');previous=m['id']
                if witnesses:hasher.update(b',')
                hasher.update(canonical(m));leaves[m['scope']].append(hashlib.sha256(b'ppcmr:leaf:v1\x00'+canonical(m)).digest())
                raw=decode(decrypt_content(client.scope_keys[m['scope']],m['feature'],aad(snap,m,'feature')))
                require(len(raw)==DIM*4,'Malformed encrypted feature')
                encoded.append(validate_feature(np.frombuffer(raw,dtype='<f4').copy()))
                witnesses.append({k:m[k] for k in ('id','scope','modality','content_digest')})
            digits.write(block['start'],signed_limb_matrix(encoded))
        require(len(witnesses)==n,'Missing cache candidates')
        for scope,part in partitions.items():require(len(leaves[scope])==part['count'] and root_from_leaves(leaves[scope])==part['root'],'Incomplete or altered feature cache')
        state=StreamLocalState(digest(client.snapshot),digest(client.credential),req['target_modality'],hasher.hexdigest(),witnesses,digits)
        return state,{'initialization_seconds':perf_counter()-start,'client_request_sent_bytes':sent,
            'client_response_received_bytes':received,'client_total_transport_payload_bytes':sent+received,
            'candidate_count':n,'signed_limb_cache_bytes':n*DIM*10*4,'block_rows':self.block_rows}

    def search_cached(self,client,session,state):
        start=perf_counter();snap,_,req=authenticate(client.owner_public,client.snapshot,client.credential,session.request)
        require(state.snapshot_digest==digest(client.snapshot) and state.credential_digest==digest(client.credential) and
            state.target_modality==req['target_modality'],'Stale/wrong-permission local cache')
        query=validate_feature(session.feature)
        sent=_send(self.connection,{'command':'prepare_cached','request':session.request,'credential':client.credential})
        response,received=self._receive()
        require(response['request_digest']==digest(session.request) and response['snapshot_digest']==state.snapshot_digest and
            response['collection_digest']==state.collection_digest and response['candidate_count']==len(state.witnesses),'Wrong cached collection binding')
        prepared_seconds=perf_counter()-start;t=perf_counter();scores=[]
        for begin in range(0,len(state.witnesses),self.block_rows):
            rows=state.digits.read(range(begin,min(begin+self.block_rows,len(state.witnesses))))
            scores.extend(exact_local_matvec(rows,query))
        order=sorted(range(len(scores)),key=lambda i:(-scores[i],state.witnesses[i]['id']))[:req['k']]
        selected=[state.witnesses[i] for i in order];scoring_seconds=perf_counter()-t
        identifier=digest(session.request);fetch=sign(client.signing_key,{'request_digest':identifier,'ids':[m['id'] for m in selected]},'float-fetch')
        t=perf_counter();sent+=_send(self.connection,{'command':'fetch','request_digest':identifier,'fetch':fetch})
        fetched,size=self._receive();received+=size
        require([r['id'] for r in fetched['results']]==[m['id'] for m in selected],'Incorrect fetched order')
        contents=[]
        for r,m in zip(fetched['results'],selected):
            require(set(r)=={'id','content'} and digest(r['content'])==m['content_digest'],'Altered content')
            contents.append(decrypt_content(client.scope_keys[m['scope']],r['content'],aad(snap,m,'content')))
        return {'accepted':True,'ids':[m['id'] for m in selected],'scores':scores,'contents':contents}, {
            'cached_request_authentication_seconds':prepared_seconds,'exact_scoring_and_ranking_seconds':scoring_seconds,
            'content_fetch_check_seconds':perf_counter()-t,'search_seconds_including_serialization':perf_counter()-start,
            'client_request_sent_bytes':sent,'client_response_received_bytes':received,
            'client_total_transport_payload_bytes':sent+received,'candidate_count':len(scores),
            'server_process_id':response['pid'],'block_rows':self.block_rows,'features_downloaded_this_query':False,
            'feature_cache_per_immutable_snapshot':True,'local_kernel':'streamed-signed-limb-fp64'}
