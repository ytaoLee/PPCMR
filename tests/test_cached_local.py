import copy
import random
import unittest

import numpy as np

from float_reference import exact_dot_encoded
from ppcmr.cached_local import CachedLocalEngine,exact_local_matvec,signed_limb_matrix
from ppcmr.crypto import digest,new_owner_key,sign
from ppcmr.float_mpc import DIM,publish
from ppcmr.local_reference import AuthorizedLocalEngine,make_local_query
from test_float_mpc import vector


class SignedLocalKernelTests(unittest.TestCase):
    def test_random_bounded_signed_values_and_extreme_carries(self):
        rng=random.Random(20261009)
        rows=[[rng.randint(-(1<<149),1<<149) for _ in range(DIM)] for _ in range(4)]
        rows += [[1<<149]*DIM,[-(1<<149)]*DIM,[0]*DIM]
        for q in ([rng.randint(-(1<<149),1<<149) for _ in range(DIM)],
                  [1<<149]*DIM,[-(1<<149)]*DIM):
            self.assertEqual(exact_local_matvec(signed_limb_matrix(rows),q),
                             [exact_dot_encoded(q,row) for row in rows])

    def test_subnormal_encoded_values_empty_and_invalid_domains(self):
        rows=[[1,-1]+[0]*(DIM-2)]
        self.assertEqual(exact_local_matvec(signed_limb_matrix(rows),[1,-1]+[0]*(DIM-2)),[2])
        self.assertEqual(exact_local_matvec(signed_limb_matrix([]),[0]*DIM),[])
        with self.assertRaises(ValueError):signed_limb_matrix([[1]])
        with self.assertRaises(ValueError):signed_limb_matrix([[1<<150]*DIM])
        with self.assertRaises(ValueError):exact_local_matvec(np.zeros((1,DIM,10)),[0]*DIM)
        with self.assertRaises(ValueError):exact_local_matvec(np.full((1,DIM,10),65536,dtype=np.int32),[0]*DIM)


class CachedLocalProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owner=new_owner_key()
        cls.records=[{'id':i,'scope':s,'modality':m,'embedding':x,'content':i}
            for i,s,m,x in [('a','s','text',vector(.75)),('b','s','text',vector(.5)),
                ('c','s','text',vector(.5)),('d','s','text',vector(-.5)),
                ('image','s','image',vector(1)),('z','t','text',vector(1))]]
        cls.server,_,cls.clients=publish(cls.records,{'u':['s'],'none':[]},{},owner_key=cls.owner,compact=True)
        cls.client=cls.clients['u'];cls.engine=CachedLocalEngine(cls.server)
        cls.state,cls.setup=cls.engine.initialize(cls.client,'image')

    @classmethod
    def tearDownClass(cls):cls.engine.close()

    def test_cached_modes_match_authenticated_download_baseline(self):
        tiny=np.nextafter(np.float32(0),np.float32(1))
        with AuthorizedLocalEngine(self.server) as download:
            for q in (vector(1),vector(-1),vector(tiny,-tiny),vector(0)):
                expected,_=download.search(self.client,make_local_query(self.client,q,'image',3))
                for kernel in ('python','signed-limb-fp64'):
                    self.engine.kernel=kernel
                    result,cost=self.engine.search_cached(self.client,make_local_query(self.client,q,'image',3),self.state)
                    self.assertEqual(result,expected)
                    self.assertFalse(cost['features_downloaded_this_query'])
                    self.assertLess(cost['client_total_transport_payload_bytes'],self.setup['client_total_transport_payload_bytes'])

    def test_empty_grants_and_reverse_direction(self):
        for client,modality in ((self.clients['none'],'image'),(self.client,'text')):
            state,_=self.engine.initialize(client,modality)
            result,_=self.engine.search_cached(client,make_local_query(client,vector(1),modality),state)
            self.assertTrue(result['accepted'])
            self.assertEqual(result['ids'],[] if client is self.clients['none'] else ['image'])

    def test_snapshot_credential_and_modality_binding(self):
        with self.assertRaisesRegex(ValueError,'Stale/wrong-permission'):
            self.engine.search_cached(self.client,make_local_query(self.client,vector(1),'text'),self.state)
        with self.assertRaisesRegex(ValueError,'Stale/wrong-permission'):
            self.engine.search_cached(self.clients['none'],make_local_query(self.clients['none'],vector(1),'image'),self.state)
        server,_,clients=publish(self.records,{'u':['s']},{},epoch=2,owner_key=self.owner,compact=True)
        with CachedLocalEngine(server) as changed:
            with self.assertRaisesRegex(ValueError,'Stale/wrong-permission'):
                changed.search_cached(clients['u'],make_local_query(clients['u'],vector(1),'image'),self.state)

    def test_replay_and_tampered_collection_acknowledgement(self):
        session=make_local_query(self.client,vector(1),'image',3)
        self.engine.search_cached(self.client,session,self.state)
        with self.assertRaisesRegex(ValueError,'Replayed'):
            self.engine.search_cached(self.client,session,self.state)
        reply={'request_digest':digest(session.request),'snapshot_digest':self.state.snapshot_digest,
               'collection_digest':'altered','candidate_count':len(self.state.witnesses)}
        with self.assertRaisesRegex(ValueError,'collection binding'):
            self.engine._validate_binding(self.client,session,self.state,reply)

    def test_initial_catalog_mutation_rejected(self):
        altered=copy.deepcopy(self.server);altered['records'].pop(0)
        with CachedLocalEngine(altered) as engine:
            with self.assertRaisesRegex(ValueError,'Incomplete or altered'):
                engine.initialize(self.client,'image')


if __name__=='__main__':unittest.main()
