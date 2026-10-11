import copy
import random
import unittest

import numpy as np

from float_reference import exact_dot
from ppcmr.cached_mpc import CachedMatrixEngine,cached_material
from ppcmr.compact_verify import prepare_verifier_state,verify_compact
from ppcmr.exact_ring import matrix_digits,matvec
from ppcmr.float_mpc import (DIM,MASK,TwoCloudEngine,_send,make_query,publish,
                            unpack_words,verify_result)
from test_float_mpc import vector


class ExactRingKernelTests(unittest.TestCase):
    def test_maximal_limbs_carries_and_ring_wraparound(self):
        rows=[[MASK]*DIM,[1]*DIM,[0]*DIM]
        result=matvec(matrix_digits(rows),[MASK]*DIM)
        self.assertEqual(result,[DIM,(-DIM)&MASK,0])

    def test_random_full_width_words_match_independent_python(self):
        rng=random.Random(20261009)
        rows=[[rng.getrandbits(320) for _ in range(DIM)] for _ in range(5)]
        for vector0 in ([rng.getrandbits(320) for _ in range(DIM)],
                        [(1<<319) if i%2 else MASK for i in range(DIM)]):
            expected=[sum(x*y for x,y in zip(row,vector0))&MASK for row in rows]
            self.assertEqual(matvec(matrix_digits(rows),vector0),expected)

    def test_invalid_domains_and_empty_matrix(self):
        with self.assertRaises(ValueError):matrix_digits([[1]])
        with self.assertRaises(ValueError):matrix_digits([[-1]*DIM])
        with self.assertRaises(ValueError):matvec(np.zeros((1,DIM,20),dtype=np.float32),[1]*DIM)
        with self.assertRaises(ValueError):matvec(matrix_digits([[0]*DIM]),[1]*(DIM-1))
        self.assertEqual(matvec(matrix_digits([]),[1]*DIM),[])

    def test_fresh_cached_correlations_match_immutable_matrix(self):
        rng=random.Random(20261010)
        matrix=[[rng.getrandbits(320) for _ in range(DIM)] for _ in range(3)]
        context={'mask':'m','snapshot':'s'}
        identifiers=[]; recovered=[]
        for kernel in ('python','limb-fp64'):
            pair=cached_material(matrix,[0,2],context,kernel,matrix_digits([matrix[0],matrix[2]]))
            b=[[v for v in unpack_words(t['b_blob'],DIM)] for t in pair]
            c=[unpack_words(t['c_blob'],2) for t in pair]
            b=[(x+y)&MASK for x,y in zip(*b)]
            c=[(x+y)&MASK for x,y in zip(*c)]
            self.assertEqual(c,[sum(x*y for x,y in zip(matrix[i],b))&MASK for i in (0,2)])
            identifiers.append(pair[0]['id']);recovered.append(b)
        self.assertNotEqual(*identifiers)
        self.assertNotEqual(*recovered)


class CachedProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records=[{'id':i,'scope':s,'modality':m,'embedding':x,'content':i}
            for i,s,m,x in [('a','s','text',vector(.75)),('b','s','text',vector(.5)),
                ('c','s','text',vector(.5)),('d','s','text',vector(-.5)),
                ('image','s','image',vector(1)),('z','t','text',vector(1))]]
        cls.server,cls.shares,cls.clients=publish(records,{'u':['s'],'none':[]},{},compact=True)
        cls.engine=CachedMatrixEngine(cls.server,cls.shares)
        cls.client=cls.clients['u'];cls.state=prepare_verifier_state(cls.client)

    @classmethod
    def tearDownClass(cls):cls.engine.close()

    def test_cached_python_and_limb_modes_match_reference_and_full_verifier(self):
        tiny=np.nextafter(np.float32(0),np.float32(1))
        with TwoCloudEngine(self.server,self.shares) as baseline:
            for feature in (vector(1),vector(-1),vector(tiny,-tiny)):
                reference,_=baseline.search(self.client,make_query(self.client,feature,'image',3),'compact')
                for kernel in ('python','limb-fp64'):
                    self.engine.kernel=kernel
                    session=make_query(self.client,feature,'image',3)
                    result,cost=self.engine.search(self.client,session,'compact')
                    self.assertEqual(result['scores'],reference['scores'])
                    self.assertEqual(result['results'],reference['results'])
                    self.assertTrue(verify_compact(self.client,session,result,self.state)['accepted'])
                    self.assertLess(cost['peer_opening_sent_bytes'],42000)
                    session=make_query(self.client,feature,'image',3)
                    result,_=self.engine.search(self.client,session)
                    self.assertTrue(verify_result(self.client,session,result)['accepted'])

    def test_zero_rights_reverse_and_maximum_score(self):
        for client,modality in ((self.clients['none'],'image'),(self.client,'text')):
            session=make_query(client,vector(1),modality)
            result,cost=self.engine.search(client,session,'compact')
            self.assertTrue(verify_compact(client,session,result)['accepted'])
            if client is self.clients['none']:
                self.assertEqual(result['scores'],[]);self.assertEqual(cost['peer_opening_sent_bytes'],0)
        maximum=np.ones(DIM,dtype=np.float32)
        records=[{'id':'max','scope':'s','modality':'text','embedding':maximum,'content':'max'}]
        server,shares,clients=publish(records,{'u':['s']},{},compact=True)
        with CachedMatrixEngine(server,shares) as engine:
            session=make_query(clients['u'],maximum,'image',1)
            result,_=engine.search(clients['u'],session,'compact')
            self.assertEqual(result['scores'],[exact_dot(maximum,maximum)])
            self.assertTrue(verify_compact(clients['u'],session,result)['accepted'])

    def test_reused_material_and_wrong_cache_context_fail_closed(self):
        original=self.engine._make_material
        first=original([m for m in self.server['catalog'] if m['scope']=='s' and m['modality']=='text'])
        try:
            self.engine._make_material=lambda _:first
            self.engine.search(self.client,make_query(self.client,vector(1),'image'),'compact')
            with self.assertRaises(ValueError):
                self.engine.search(self.client,make_query(self.client,vector(1),'image'),'compact')
            wrong=copy.deepcopy(original([m for m in self.server['catalog'] if m['scope']=='s' and m['modality']=='text']))
            for t in wrong:t['mask']='wrong'
            self.engine._make_material=lambda _:wrong
            with self.assertRaises(ValueError):
                self.engine.search(self.client,make_query(self.client,vector(1),'image'),'compact')
        finally:self.engine._make_material=original

    def test_cache_cannot_be_reset_or_silently_rebound_to_new_snapshot(self):
        for connection in self.engine.connections:_send(connection,{'command':'cache_matrix'})
        responses=self.engine._responses()
        self.assertTrue(all(not r['ok'] and 'already initialized' in r['error'] for r in responses))
        old=self.engine.server_data['snapshot']
        try:
            self.engine.server_data['snapshot']=copy.deepcopy(old)
            self.engine.server_data['snapshot']['payload']['epoch']+=1
            with self.assertRaisesRegex(ValueError,'new workers'):
                self.engine.search(self.client,make_query(self.client,vector(1),'image'),'compact')
        finally:self.engine.server_data['snapshot']=old

    def test_query_authentication_and_response_verification_preserved(self):
        session=make_query(self.client,vector(1),'image',3)
        session.shares[0][0]^=1
        with self.assertRaisesRegex(ValueError,'commitment'):
            self.engine.search(self.client,session,'compact')
        session=make_query(self.client,vector(1),'image',3)
        result,_=self.engine.search(self.client,session,'compact')
        result['scores'][0]+=1
        self.assertFalse(verify_compact(self.client,session,result,self.state)['accepted'])
        with self.assertRaisesRegex(ValueError,'Replayed'):
            self.engine.search(self.client,session,'compact')


if __name__=='__main__':unittest.main()
