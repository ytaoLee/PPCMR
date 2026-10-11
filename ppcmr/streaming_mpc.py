"""Exact two-cloud MPC with bounded row windows and actual IPC openings."""
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import struct
import time
from time import perf_counter
import numpy as np
from .crypto import canonical,digest,sign,load_public_key,verify_signature
from .float_mpc import (DIM,MASK,MODULUS,HALF,TwoCloudEngine,_recv,_send,_recv_counted,
    authenticate,authenticate_query_share,prepare_message,require,random_words,split_values,pack_words,unpack_words)
from .compact_verify import validate_catalog,FORMAT
from .publication_policy import validate_publication
from .exact_ring import matvec
from .streaming_store import RingFile,random_digits,ring_combine,digits_words


def exchange(peer,party,binding,own):
    header=canonical(binding);message=struct.pack('<I',len(header))+header+own
    if party==0:peer.send_bytes(message);other=peer.recv_bytes()
    else:other=peer.recv_bytes();peer.send_bytes(message)
    require(len(other)>=4,'Truncated opening');size=struct.unpack('<I',other[:4])[0]
    require(json.loads(other[4:4+size])==binding,'MPC opening context differs')
    body=other[4+size:];require(len(body)==len(own),'Opening dimensions differ')
    return body,len(message)


def _stream_worker(party,control,peer,server,xfile,mode,folder,block_rows):
    used_nonces=set();used_triples=set();prepared={};sessions={};cache=None;active=None
    D=RingFile(Path(folder)/f'D_{party}.bin',len(server['records']),create=True) if mode=='cached' else None
    try:
        while True:
            message=_recv(control);command=message.get('command')
            if command=='stop':break
            try:
                if command=='cache_begin':
                    context=message['context']
                    require(mode=='cached' and cache is None,'Matrix cache already initialized or disabled')
                    require(context['snapshot']==digest(server['snapshot']) and
                        context['record_order']==digest([m['id'] for m in server['records']]) and
                        context['rows']==len(server['records']) and context['dimensions']==DIM and
                        context['ring_bits']==320 and context['stage']=='streamed-immutable-matrix.v1',
                        'Matrix cache version/order mismatch')
                    cache={'context':context,'cursor':0,'hasher':hashlib.sha256(),'ready':False,'peer_bytes':0}
                    _send(control,{'ok':True,'pid':os.getpid()})
                elif command=='cache_block':
                    require(cache is not None and not cache['ready'],'Cache unavailable or already initialized')
                    start,count=message['start'],message['count']
                    require(start==cache['cursor'] and 0<count<=block_rows and start+count<=D.rows,'Cache block order/dimensions differ')
                    blob=message['A_blob'];require(len(blob)==count*DIM*40,'Wrong mask block size')
                    mask=np.frombuffer(blob,dtype='<u2').reshape(count,DIM,20)
                    own=ring_combine(xfile.read(range(start,start+count)),mask,True)
                    binding={**cache['context'],'block_start':start,'block_rows':count}
                    other,sent=exchange(peer,party,binding,own.tobytes())
                    opened=ring_combine(own,np.frombuffer(other,dtype='<u2').reshape(own.shape))
                    D.write(start,opened);cache['hasher'].update(opened.tobytes())
                    cache['cursor']+=count;cache['peer_bytes']+=sent
                    _send(control,{'ok':True})
                elif command=='cache_finish':
                    require(cache is not None and not cache['ready'] and cache['cursor']==D.rows,'Incomplete matrix cache')
                    cache['ready']=True
                    _send(control,{'ok':True,'opened_digest':cache['hasher'].hexdigest(),'peer_sent_bytes':cache['peer_bytes'],'pid':os.getpid()})
                elif command=='prepare':
                    snap,cred,req=authenticate(server['owner_public'],server['snapshot'],message['credential'],message['request'])
                    require(req['nonce'] not in used_nonces,'Replayed query')
                    authenticate_query_share(cred,message['request'],message['share_envelope'],party,message['share'])
                    used_nonces.add(req['nonce']);identifier=digest(message['request'])
                    indices=[i for i,m in enumerate(server['records']) if m['scope'] in cred['allowed_scopes'] and m['modality']==req['target_modality']]
                    prepared[identifier]=(indices,message['share'],req)
                    sessions={identifier:{'indices':indices,'credential':message['credential']}}
                    witnesses=[server['catalog'][i] for i in indices]
                    _send(control,{'ok':True,'ids':[m['id'] for m in witnesses],'witnesses':witnesses if party==0 else None})
                elif command=='abort':
                    prepared.pop(message['request_digest'],None);active=None
                    _send(control,{'ok':True})
                elif command=='begin':
                    identifier=message['request_digest'];indices,q,req=prepared.pop(identifier)
                    context=message['context']
                    require(context['request']==identifier and context['snapshot']==digest(server['snapshot']) and
                        context['candidates']==digest(indices) and context['mode']==mode,'Wrong preprocessing context')
                    require(context['triple'] not in used_triples,'Reused preprocessing triple')
                    if mode=='cached':require(cache and cache['ready'] and context['mask']==cache['context']['mask'],'Wrong matrix-cache context')
                    used_triples.add(context['triple'])
                    b=unpack_words(message['b_blob'],DIM if indices else 0)
                    started=perf_counter();peer_bytes=0;e=[]
                    if indices:
                        own=[(x-y)&MASK for x,y in zip(q,b)]
                        other,peer_bytes=exchange(peer,party,{**context,'stage':'streamed-query-open.v1'},pack_words(own))
                        e=[(x+y)&MASK for x,y in zip(own,unpack_words(other,DIM))]
                    active={'context':context,'indices':indices,'b':b,'e':e,'cursor':0,'scores':[],
                            'peer_bytes':peer_bytes,'cpu_seconds':perf_counter()-started}
                    _send(control,{'ok':True})
                elif command=='block':
                    require(active is not None and message['request_digest']==active['context']['request'],'No bound active query')
                    start,count=message['start'],message['count'];indices=active['indices']
                    require(start==active['cursor'] and 0<count<=block_rows and start+count<=len(indices),'Query block order/dimensions differ')
                    selected=indices[start:start+count];c=unpack_words(message['c_blob'],count)
                    t=perf_counter();x=xfile.read(selected);b=active['b'];e=active['e']
                    if mode=='cached':
                        values=[(a+v+w)&MASK for a,v,w in zip(c,matvec(D.read(selected),b),matvec(x,e))]
                    else:
                        blob=message['A_blob'];require(len(blob)==count*DIM*40,'Wrong fresh matrix block')
                        a=np.frombuffer(blob,dtype='<u2').reshape(count,DIM,20)
                        own=ring_combine(x,a,True)
                        binding={**active['context'],'stage':'streamed-fresh-matrix-open.v1','block_start':start,'block_rows':count}
                        other,sent=exchange(peer,party,binding,own.tobytes());active['peer_bytes']+=sent
                        opened=ring_combine(own,np.frombuffer(other,dtype='<u2').reshape(own.shape))
                        # Preserve the original fresh-MPC Python arithmetic baseline.
                        av=digits_words(a);dv=digits_words(opened);values=[]
                        for row in range(count):
                            aa=av[row*DIM:(row+1)*DIM];dd=dv[row*DIM:(row+1)*DIM]
                            value=c[row]+sum(v*w+y*z for v,w,y,z in zip(dd,b,aa,e))
                            if party==0:value+=sum(v*w for v,w in zip(dd,e))
                            values.append(value&MASK)
                    active['cpu_seconds']+=perf_counter()-t;active['scores'].extend(values);active['cursor']+=count
                    _send(control,{'ok':True})
                elif command=='finish':
                    require(active is not None and message['request_digest']==active['context']['request'] and
                            active['cursor']==len(active['indices']),'Incomplete query computation')
                    _send(control,{'ok':True,'request_digest':active['context']['request'],'shares':active['scores'],
                        'seconds':active['cpu_seconds'],'peer_sent_bytes':active['peer_bytes'],'pid':os.getpid()});active=None
                elif command=='fetch':
                    identifier=message['request_digest'];stored=sessions[identifier]
                    cred=verify_signature(load_public_key(server['owner_public']),stored['credential'],'float-credential')
                    require(cred['expires_at']>time.time(),'Expired credential')
                    payload=verify_signature(load_public_key(cred['query_public']),message['fetch'],'float-fetch')
                    require(set(payload)=={'request_digest','ids'} and payload['request_digest']==identifier,'Wrong fetch request')
                    ids=payload['ids'];allowed={server['records'][i]['id'] for i in stored['indices']}
                    require(isinstance(ids,list) and len(ids)==len(set(ids)) and set(ids).issubset(allowed),'Unauthorized content fetch')
                    _send(control,{'ok':True,'results':[{'id':i,'content':server['contents'][i]} for i in ids]})
                else:raise ValueError('Unknown streamed MPC command')
            except Exception as error:_send(control,{'ok':False,'error':str(error)})
    finally:
        server['contents'].close();server['metadata_store'].close();control.close();peer.close()


class StreamingEngine(TwoCloudEngine):
    def __init__(self,server_data,shares,folder,mode='cached',block_rows=256):
        require(mode in {'fresh','cached'} and 1<=block_rows<=4096,'Invalid streamed engine')
        snap=verify_signature(load_public_key(server_data['owner_public']),server_data['snapshot'],'float-snapshot');validate_publication(snap)
        self.server_data=server_data;self._snapshot=digest(server_data['snapshot']);self.mode=mode;self.block_rows=block_rows
        self.positions={m['id']:i for i,m in enumerate(server_data['records'])}
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        ctx=mp.get_context('spawn');peer0,peer1=ctx.Pipe(duplex=True);self.connections=[];self.processes=[]
        start=perf_counter()
        for party,peer in enumerate((peer0,peer1)):
            parent,child=ctx.Pipe(duplex=True)
            p=ctx.Process(target=_stream_worker,args=(party,child,peer,server_data,shares[party],mode,str(self.folder),block_rows))
            p.start();child.close();self.connections.append(parent);self.processes.append(p)
        peer0.close();peer1.close();self.startup_seconds=perf_counter()-start;self.received_bytes=0
        self.matrix_context=None;self.setup_metrics={}
        if mode=='cached':self._setup_cache()

    def _responses(self):
        result=[]
        for connection in self.connections:
            require(connection.poll(180),'Cloud response timeout')
            body,size=_recv_counted(connection);self.received_bytes+=size;result.append(body)
        return result

    def _checked(self):
        result=self._responses();require(all(r['ok'] for r in result),'; '.join(r.get('error','') for r in result if not r['ok']))
        return result

    def _setup_cache(self):
        start=perf_counter();rows=len(self.server_data['records'])
        self.A=RingFile(self.folder/'dealer_A.bin',rows,create=True)
        self.matrix_context={'stage':'streamed-immutable-matrix.v1','mask':os.urandom(16).hex(),'snapshot':self._snapshot,
            'record_order':digest(list(self.positions)),'rows':rows,'dimensions':DIM,'ring_bits':320}
        sent=0;generation=0.
        for connection in self.connections:sent+=_send(connection,{'command':'cache_begin','context':self.matrix_context})
        self._checked()
        for begin in range(0,rows,self.block_rows):
            count=min(self.block_rows,rows-begin);t=perf_counter();a=random_digits(count);first=random_digits(count)
            second=ring_combine(a,first,True);self.A.write(begin,a);generation+=perf_counter()-t
            for connection,own in zip(self.connections,(first,second)):
                sent+=_send(connection,{'command':'cache_block','start':begin,'count':count,'A_blob':own.tobytes()})
            self._checked()
        for connection in self.connections:sent+=_send(connection,{'command':'cache_finish'})
        responses=self._checked();require(responses[0]['opened_digest']==responses[1]['opened_digest'],'Cloud masked matrices disagree')
        self.setup_metrics={'matrix_setup_seconds':perf_counter()-start,'matrix_mask_generation_seconds':generation,
            'dealer_setup_sent_bytes':sent,'peer_setup_sent_bytes':sum(r['peer_sent_bytes'] for r in responses),
            'control_setup_received_bytes':self.received_bytes,'records':rows,'block_rows':self.block_rows,
            'one_mask_per_immutable_snapshot':True,'fresh_b_and_c_shares_per_query':True}

    def search(self,client,session,witness_mode='compact'):
        require(witness_mode=='compact','Streamed scale engine uses the compact verifier')
        require(digest(self.server_data['snapshot'])==self._snapshot,'Immutable cache requires new workers for a new snapshot')
        self.received_bytes=0;identifier=digest(session.request);start=perf_counter();client_sent=0
        for party,connection in enumerate(self.connections):client_sent+=_send(connection,prepare_message(client,session,party))
        responses=self._responses()
        if not all(r['ok'] for r in responses):
            for connection in self.connections:_send(connection,{'command':'abort','request_digest':identifier})
            self._responses();raise ValueError('; '.join(r.get('error','') for r in responses if not r['ok']))
        witnesses=responses[0]['witnesses'];validate_catalog(client,session,witnesses)
        ids=[m['id'] for m in witnesses];require(all(r['ids']==ids for r in responses),'Cloud candidate sets disagree')
        indices=[self.positions[i] for i in ids];prepare_seconds=perf_counter()-start
        t=perf_counter();b=random_words(DIM) if indices else [];b0,b1=split_values(b)
        context={'request':identifier,'snapshot':self._snapshot,'triple':os.urandom(16).hex(),
                 'candidates':digest(indices),'mode':self.mode}
        if self.mode=='cached':context['mask']=self.matrix_context['mask']
        preprocessing=perf_counter()-t;dealer_sent=0
        online_start=perf_counter()
        for connection,own in zip(self.connections,(b0,b1)):
            dealer_sent+=_send(connection,{'command':'begin','request_digest':identifier,'context':context,'b_blob':pack_words(own)})
        self._checked()
        for begin in range(0,len(indices),self.block_rows):
            selected=indices[begin:begin+self.block_rows];t=perf_counter()
            if self.mode=='cached':
                c=matvec(self.A.read(selected),b);a_pair=(None,None)
            else:
                a=random_digits(len(selected));words=digits_words(a)
                c=[sum(v*w for v,w in zip(words[row*DIM:(row+1)*DIM],b))&MASK for row in range(len(selected))]
                first=random_digits(len(selected));a_pair=(first,ring_combine(a,first,True))
            c_pair=split_values(c);preprocessing+=perf_counter()-t
            for connection,own,cc in zip(self.connections,a_pair,c_pair):
                message={'command':'block','request_digest':identifier,'start':begin,'count':len(selected),'c_blob':pack_words(cc)}
                if own is not None:message['A_blob']=own.tobytes()
                dealer_sent+=_send(connection,message)
            self._checked()
        for connection in self.connections:dealer_sent+=_send(connection,{'command':'finish','request_digest':identifier})
        computed=self._checked();require(all(r['request_digest']==identifier and len(r['shares'])==len(ids) for r in computed),'Incomplete score response')
        scores=[]
        for a,b0 in zip(computed[0]['shares'],computed[1]['shares']):
            require(type(a) is int and type(b0) is int and 0<=a<=MASK and 0<=b0<=MASK,'Invalid output share')
            value=(a+b0)&MASK;scores.append(value-MODULUS if value>=HALF else value)
        order=sorted(range(len(ids)),key=lambda i:(-scores[i],ids[i]))[:min(session.request['payload']['k'],len(ids))]
        selected=[ids[i] for i in order];online_seconds=perf_counter()-online_start
        fetch=sign(client.signing_key,{'request_digest':identifier,'ids':selected},'float-fetch')
        client_sent+=_send(self.connections[0],{'command':'fetch','request_digest':identifier,'fetch':fetch})
        require(self.connections[0].poll(180),'Content fetch timeout');fetched,size=_recv_counted(self.connections[0]);self.received_bytes+=size
        require(fetched['ok'],'Content fetch denied')
        result={'request_digest':identifier,'snapshot_digest':self._snapshot,'witnesses':witnesses,'scores':scores,
                'results':fetched['results'],'proof_format':FORMAT}
        return result,{'prepare_seconds':prepare_seconds,'trusted_preprocessing_seconds':preprocessing,
            'online_seconds_including_triple_delivery':online_seconds,
            'end_to_end_seconds_without_verification':perf_counter()-start,
            'client_request_sent_bytes':client_sent,'trusted_preprocessing_sent_bytes':dealer_sent,
            'client_response_received_bytes':self.received_bytes,'client_total_transport_payload_bytes':client_sent+self.received_bytes,
            'peer_opening_sent_bytes':sum(r['peer_sent_bytes'] for r in computed),
            'server_cpu_phase_seconds':[r['seconds'] for r in computed],'cloud_process_ids':[r['pid'] for r in computed],
            'candidate_count':len(ids),'topk_location':'client','block_rows':self.block_rows,
            'mode':self.mode,'timing_note':'Online interval includes interleaved preprocessing; do not add its component time again.'}
