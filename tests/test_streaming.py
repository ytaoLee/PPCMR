import copy
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
from float_reference import encode_fp32,exact_dot
from ppcmr.streaming_store import RingFile,random_digits,ring_combine,digits_words,publish_streamed,root_from_leaves
from ppcmr.streaming_mpc import StreamingEngine
from ppcmr.streaming_local import StreamingLocalEngine
from ppcmr.float_mpc import DIM,MASK,make_query,_send,prepare_message,pack_words
from ppcmr.local_reference import make_local_query
from ppcmr.compact_verify import prepare_verifier_state,verify_compact
from ppcmr.crypto import merkle_root,digest


class Records:
    def __init__(self,rows):self.rows=sorted(rows,key=lambda r:r['id']);self.descriptors=self.rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[i]


def vector(value):
    out=np.zeros(DIM,dtype=np.float32);out[0]=value;return out


class StreamingTests(unittest.TestCase):
    def test_ring_carries_borrows_and_noncontiguous_file_windows(self):
        rng=random.Random(8123)
        values=[rng.getrandbits(320) for _ in range(3*DIM)]
        other=[MASK if i%2 else 1 for i in range(3*DIM)]
        def digits(words):return np.frombuffer(b''.join(v.to_bytes(40,'little') for v in words),dtype='<u2').reshape(3,DIM,20)
        a,b=digits(values),digits(other)
        self.assertEqual(digits_words(ring_combine(a,b)),[(x+y)&MASK for x,y in zip(values,other)])
        self.assertEqual(digits_words(ring_combine(a,b,True)),[(x-y)&MASK for x,y in zip(values,other)])
        with tempfile.TemporaryDirectory() as folder:
            file=RingFile(Path(folder)/'ring.bin',3,create=True);file.write(0,a)
            self.assertEqual(digits_words(file.read([2,0])),values[2*DIM:]+values[:DIM])
            with self.assertRaises(ValueError):file.read([-1])
        members=[{'id':str(i),'test':i} for i in range(5)]
        import hashlib
        from ppcmr.crypto import canonical
        leaves=[hashlib.sha256(b'ppcmr:leaf:v1\x00'+canonical(m)).digest() for m in members]
        self.assertEqual(root_from_leaves(leaves),merkle_root(members))

    def fixture(self,folder):
        tiny=np.nextafter(np.float32(0),np.float32(1))
        rows=[{'id':name,'scope':scope,'modality':modality,'embedding':vector(value),'content':name}
            for name,scope,modality,value in [('a','s','text',.5),('b','s','text',.5),('c','s','text',-.5),
                ('d','s','text',tiny),('e','s','image',1),('f','t','text',1),('g','t','image',-1)]]
        records=Records(rows)
        server,shares,clients,_=publish_streamed(records,{'u':['s'],'all':['s','t'],'none':[]},Path(folder)/'publication',2)
        for party in (0,1):self.assertEqual(shares[party].rows,len(records))
        actual=digits_words(ring_combine(shares[0].read(range(len(records))),shares[1].read(range(len(records)))))
        self.assertEqual(actual,[v&MASK for r in records.rows for v in encode_fp32(r['embedding'])])
        for part in server['snapshot']['payload']['partitions']:
            metadata=[server['metadata_store'][r['id']] for r in records.rows if r['scope']==part['scope'] and r['modality']==part['modality']]
            self.assertEqual(merkle_root(metadata),part['root'])
        return records,server,shares,clients

    def test_fresh_cached_and_local_match_independent_exact_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            records,server,shares,clients=self.fixture(folder)
            try:
                for mode in ('fresh','cached'):
                    with StreamingEngine(server,shares,Path(folder)/mode,mode,2) as engine:
                        for subject in ('u','all','none'):
                            client=clients[subject];state=prepare_verifier_state(client)
                            for modality,feature in (('image',vector(1)),('image',vector(-1)),
                                                     ('image',vector(np.nextafter(np.float32(0),np.float32(1)))),('text',vector(1))):
                                candidates=[r for r in records.rows if r['scope'] in client.scope_keys and r['modality']!=modality]
                                scores=[exact_dot(feature,r['embedding']) for r in candidates]
                                order=sorted(range(len(scores)),key=lambda i:(-scores[i],candidates[i]['id']))[:3]
                                session=make_query(client,feature,modality,3);result,_=engine.search(client,session)
                                self.assertEqual(result['scores'],scores)
                                self.assertEqual([r['id'] for r in result['results']],[candidates[i]['id'] for i in order])
                                self.assertTrue(verify_compact(client,session,result,state)['accepted'])
                                if scores:
                                    bad=copy.deepcopy(result);bad['scores'][0]+=1
                                    self.assertFalse(verify_compact(client,session,bad,state)['accepted'])
                                    bad=copy.deepcopy(result);bad['witnesses'].pop();bad['scores'].pop()
                                    self.assertFalse(verify_compact(client,session,bad,state)['accepted'])
                with StreamingLocalEngine(server,Path(folder)/'local',2) as engine:
                    for subject in ('u','none'):
                        client=clients[subject]
                        for modality in ('image','text'):
                            state,_=engine.initialize(client,modality)
                            feature=vector(-1);session=make_local_query(client,feature,modality,3)
                            result,_=engine.search_cached(client,session,state)
                            candidates=[r for r in records.rows if r['scope'] in client.scope_keys and r['modality']!=modality]
                            scores=[exact_dot(feature,r['embedding']) for r in candidates]
                            self.assertEqual(result['scores'],scores)
            finally:server['contents'].close();server['metadata_store'].close()

    def test_share_binding_replay_and_cache_rebinding_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            records,server,shares,clients=self.fixture(folder);client=clients['u']
            try:
                with StreamingEngine(server,shares,Path(folder)/'cached','cached',2) as engine:
                    session=make_query(client,vector(1),'image');session.shares[0][0]^=1
                    with self.assertRaisesRegex(ValueError,'commitment'):engine.search(client,session)
                    session=make_query(client,vector(1),'image');engine.search(client,session)
                    with self.assertRaisesRegex(ValueError,'Replayed'):engine.search(client,session)
                    for connection in engine.connections:_send(connection,{'command':'cache_begin','context':engine.matrix_context})
                    responses=engine._responses();self.assertTrue(all(not r['ok'] for r in responses))
                    original=server['snapshot'];server['snapshot']=copy.deepcopy(original);server['snapshot']['payload']['epoch']+=1
                    with self.assertRaisesRegex(ValueError,'new workers'):engine.search(client,make_query(client,vector(1),'image'))
                    server['snapshot']=original
            finally:server['contents'].close();server['metadata_store'].close()

    def test_incomplete_out_of_order_and_reused_correlations_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            records,server,shares,clients=self.fixture(folder);client=clients['u']
            try:
                with StreamingEngine(server,shares,Path(folder)/'cached','cached',2) as engine:
                    session=make_query(client,vector(1),'image');identifier=digest(session.request)
                    for party,connection in enumerate(engine.connections):_send(connection,prepare_message(client,session,party))
                    prepared=engine._checked();ids=prepared[0]['ids'];indices=[engine.positions[i] for i in ids]
                    context={'request':identifier,'snapshot':engine._snapshot,'triple':'fixed-test-correlation',
                        'candidates':digest(indices),'mode':'cached','mask':engine.matrix_context['mask']}
                    for connection in engine.connections:_send(connection,{'command':'begin','request_digest':identifier,'context':context,'b_blob':pack_words([0]*DIM)})
                    engine._checked()
                    for connection in engine.connections:_send(connection,{'command':'block','request_digest':identifier,'start':1,'count':1,'c_blob':pack_words([0])})
                    self.assertTrue(all(not r['ok'] for r in engine._responses()))
                    for connection in engine.connections:_send(connection,{'command':'finish','request_digest':identifier})
                    self.assertTrue(all(not r['ok'] for r in engine._responses()))
                    next_session=make_query(client,vector(1),'image');next_id=digest(next_session.request)
                    for party,connection in enumerate(engine.connections):_send(connection,prepare_message(client,next_session,party))
                    engine._checked();context['request']=next_id
                    for connection in engine.connections:_send(connection,{'command':'begin','request_digest':next_id,'context':context,'b_blob':pack_words([0]*DIM)})
                    self.assertTrue(all(not r['ok'] and 'Reused' in r['error'] for r in engine._responses()))
                    result,_=engine.search(client,make_query(client,vector(1),'image'))
                    self.assertEqual(result['scores'],[exact_dot(vector(1),r['embedding']) for r in records.rows if r['scope']=='s' and r['modality']=='text'])
            finally:server['contents'].close();server['metadata_store'].close()


if __name__=='__main__':unittest.main()
