import copy
import unittest

import numpy as np

from float_reference import exact_dot
from ppcmr.crypto import digest, new_owner_key, sign
from ppcmr.local_reference import AuthorizedLocalEngine, make_local_query
from ppcmr.float_mpc import (DIM, MASK, MODULUS, TwoCloudEngine, expected_scores,
    authenticate_query_share, make_query, make_triple, publish, share_digest,
    validate_feature, verify_result)


def vector(first, second=0):
    result = np.zeros(DIM,dtype=np.float32)
    result[:2] = [first,second]
    return result


class FloatArithmeticTests(unittest.TestCase):
    def test_negative_subnormal_and_exact_fp32_rounding_counterexample(self):
        q = vector(1,2**-12)
        a,b = vector(1,0),vector(1,2**-12)
        self.assertEqual(q @ a,q @ b)
        self.assertGreater(exact_dot(q,b),exact_dot(q,a))
        tiny = np.nextafter(np.float32(0),np.float32(1))
        self.assertEqual(exact_dot(vector(tiny),vector(-tiny)),-1)

    def test_finite_domain_and_no_ring_wraparound(self):
        with self.assertRaises(ValueError):
            validate_feature(np.ones(DIM,dtype=np.float64))
        with self.assertRaises(ValueError):
            validate_feature(vector(np.inf))
        with self.assertRaises(ValueError):
            validate_feature(vector(1.1))
        maximum = np.ones(DIM,dtype=np.float32)
        self.assertEqual(exact_dot(maximum,maximum),2**307)
        self.assertLess(exact_dot(maximum,maximum),MODULUS//2)

    def test_fresh_matrix_vector_triples_are_correlated_correctly(self):
        a,b = make_triple(3)
        A = [[(x+y)&MASK for x,y in zip(r,s)] for r,s in zip(a['A'],b['A'])]
        B = [(x+y)&MASK for x,y in zip(a['b'],b['b'])]
        C = [(x+y)&MASK for x,y in zip(a['c'],b['c'])]
        self.assertEqual(C,[sum(x*y for x,y in zip(r,B))&MASK for r in A])
        self.assertNotEqual(a['id'],make_triple(1)[0]['id'])


class FloatProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records = [
            {'id':'a','scope':'allowed','modality':'text','embedding':vector(.75),'content':'best'},
            {'id':'b','scope':'allowed','modality':'text','embedding':vector(.5),'content':'tie-first'},
            {'id':'c','scope':'allowed','modality':'text','embedding':vector(.5),'content':'tie-second'},
            {'id':'d','scope':'allowed','modality':'text','embedding':vector(-.5),'content':'negative'},
            {'id':'z','scope':'blocked','modality':'text','embedding':vector(1),'content':'unauthorized'},
            {'id':'image','scope':'allowed','modality':'image','embedding':vector(1),'content':'image-bytes'},
        ]
        server,shares,cls.clients = publish(records,{'alice':['allowed'],'none':[]},{'test':True})
        cls.engine = TwoCloudEngine(server,shares)
        cls.client = cls.clients['alice']
        cls.session = make_query(cls.client,vector(1),'image',3)
        cls.result,cls.cost = cls.engine.search(cls.client,cls.session)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_two_real_worker_processes_exact_scores_and_authorized_ties(self):
        self.assertEqual(len(set(self.cost['cloud_process_ids'])),2)
        verified = verify_result(self.client,self.session,self.result)
        self.assertTrue(verified['accepted'],verified)
        self.assertEqual(verified['ids'],['a','b','c'])
        self.assertEqual(verified['contents'],['best','tie-first','tie-second'])
        self.assertEqual(self.result['scores'],expected_scores(self.client,self.session,self.result['witnesses']))
        self.assertEqual(len(self.result['witnesses']),4)
        self.assertNotIn('z',verified['ids'])

    def test_zero_permissions_and_reverse_direction(self):
        for client,modality,expected in ((self.clients['none'],'image',[]),(self.client,'text',['image'])):
            session = make_query(client,vector(1),modality,10)
            result,_ = self.engine.search(client,session)
            verified = verify_result(client,session,result)
            self.assertTrue(verified['accepted'],verified)
            self.assertEqual(verified['ids'],expected)

    def test_server_rejects_replayed_query_and_bad_signature(self):
        with self.assertRaisesRegex(ValueError,'Replayed'):
            self.engine.search(self.client,self.session)
        session = make_query(self.client,vector(1),'image',3)
        session.request['payload']['k'] = 4
        with self.assertRaises(ValueError):
            self.engine.search(self.client,session)

    def test_reject_wrong_share_before_peer_exchange(self):
        session = make_query(self.client,vector(1),'image',3)
        session.shares[0][0] ^= 1
        with self.assertRaisesRegex(ValueError,'commitment'):
            self.engine.search(self.client,session)

    def test_private_envelope_rejects_recipient_swap_and_session_mix(self):
        for mutation in ('recipient', 'session', 'unsigned'):
            with self.subTest(mutation=mutation):
                session = make_query(self.client,vector(1),'image',3)
                envelopes = list(session.share_envelopes)
                if mutation == 'recipient':
                    envelopes.reverse()
                elif mutation == 'session':
                    envelopes[0] = make_query(self.client,vector(1),'image',3).share_envelopes[0]
                else:
                    envelopes[0] = copy.deepcopy(envelopes[0])
                    envelopes[0]['payload']['share_digest'] = '0'*64
                session.share_envelopes = tuple(envelopes)
                with self.assertRaises(ValueError):
                    self.engine.search(self.client,session)

    def test_public_request_has_no_query_dependent_share_bindings(self):
        session = make_query(self.client,vector(1),'image',3)
        self.assertEqual(set(session.request['payload']), {
            'schema','subject','credential_digest','snapshot_digest',
            'query_modality','target_modality','k','nonce'})
        self.assertEqual(session.request['payload']['schema'],'ppcmr.float-mpc.query.v2')
        for party in (0,1):
            binding = authenticate_query_share(self.client.credential['payload'],
                session.request,session.share_envelopes[party],party,session.shares[party])
            self.assertEqual(binding['share_digest'],share_digest(session.shares[party]))
            self.assertNotIn(share_digest(session.shares[1-party]),str(session.request))
            self.assertNotIn(share_digest(session.shares[1-party]),str(session.share_envelopes[party]))

    def test_legacy_public_commitment_request_is_rejected(self):
        session = make_query(self.client,vector(1),'image',3)
        payload = dict(session.request['payload'])
        payload['schema'] = 'ppcmr.float-mpc.query.v1'
        payload['share_commitments'] = [share_digest(s) for s in session.shares]
        session.request = sign(self.client.signing_key,payload,'float-query')
        with self.assertRaisesRegex(ValueError,'Malformed query'):
            self.engine.search(self.client,session)

    def test_response_tampering_and_incomplete_topk(self):
        variants = []
        for kind in ('score','missing_score','missing_witness','duplicate_witness',
                     'scope','feature_ciphertext','missing_result','order','content','version','request'):
            bad = copy.deepcopy(self.result)
            if kind == 'score': bad['scores'][0] += 1
            if kind == 'missing_score': bad['scores'].pop()
            if kind == 'missing_witness': bad['witnesses'].pop()
            if kind == 'duplicate_witness': bad['witnesses'].append(copy.deepcopy(bad['witnesses'][0]))
            if kind == 'scope': bad['witnesses'][0]['scope'] = 'blocked'
            if kind == 'feature_ciphertext': bad['witnesses'][0]['feature']['ciphertext'] = 'AAAA'
            if kind == 'missing_result': bad['results'].pop()
            if kind == 'order': bad['results'][0],bad['results'][1] = bad['results'][1],bad['results'][0]
            if kind == 'content': bad['results'][0]['content']['ciphertext'] = 'AAAA'
            if kind == 'version': bad['snapshot_digest'] = '0'*64
            if kind == 'request': bad['request_digest'] = '0'*64
            variants.append((kind,bad))
        for kind,bad in variants:
            with self.subTest(kind=kind):
                self.assertFalse(verify_result(self.client,self.session,bad)['accepted'])

    def test_valid_lower_ranked_record_cannot_replace_best(self):
        bad = copy.deepcopy(self.result)
        bad['results'] = [bad['results'][1],bad['results'][2],
            {'id':'d','content':self.engine.server_data['contents']['d']}]
        self.assertFalse(verify_result(self.client,self.session,bad)['accepted'])

    def test_private_query_is_bound_to_signed_shares(self):
        session = copy.copy(self.session)
        session.feature = vector(-1)
        self.assertFalse(verify_result(self.client,session,self.result)['accepted'])

    def test_client_signed_wrong_version_is_rejected(self):
        session = make_query(self.client,vector(1),'image',3)
        session.request['payload']['snapshot_digest'] = '0'*64
        session.request = sign(self.client.signing_key,session.request['payload'],'float-query')
        with self.assertRaisesRegex(ValueError,'version'):
            self.engine.search(self.client,session)

    def test_forged_scope_grant_is_rejected(self):
        client = copy.copy(self.client)
        client.credential = copy.deepcopy(self.client.credential)
        client.credential['payload']['allowed_scopes'] = ['allowed','blocked']
        session = make_query(client,vector(1),'image',3)
        with self.assertRaises(ValueError):
            self.engine.search(client,session)

    def test_expired_owner_signed_credential_is_rejected(self):
        owner = new_owner_key()
        records = [{'id':'record','scope':'s','modality':'text','embedding':vector(1),'content':'x'}]
        server,shares,clients = publish(records,{'alice':['s']},{},owner_key=owner)
        client = clients['alice']
        client.credential['payload']['expires_at'] = 0
        client.credential = sign(owner,client.credential['payload'],'float-credential')
        with TwoCloudEngine(server,shares) as engine:
            with self.assertRaisesRegex(ValueError,'Expired'):
                engine.search(client,make_query(client,vector(1),'image',1))

    def test_epoch_revocation_blocks_old_credential_with_same_owner_key(self):
        owner = new_owner_key()
        records = [{'id':'record','scope':'s','modality':'text','embedding':vector(1),'content':'x'}]
        _,_,old = publish(records,{'alice':['s']},{},epoch=1,owner_key=owner)
        server,shares,new = publish(records,{'alice':[]},{},epoch=2,owner_key=owner)
        with TwoCloudEngine(server,shares) as engine:
            with self.assertRaisesRegex(ValueError,'Stale'):
                engine.search(old['alice'],make_query(old['alice'],vector(1),'image',1))
            session = make_query(new['alice'],vector(1),'image',1)
            result,_ = engine.search(new['alice'],session)
            self.assertTrue(verify_result(new['alice'],session,result)['accepted'])
            self.assertEqual(result['results'],[])

    def test_authorized_local_baseline_matches_exact_verified_results(self):
        with AuthorizedLocalEngine(self.engine.server_data) as engine:
            session = make_local_query(self.client,vector(1),'image',3)
            result,cost = engine.search(self.client,session)
            self.assertEqual(result['scores'],self.result['scores'])
            self.assertEqual(result['ids'],['a','b','c'])
            self.assertEqual(result['contents'],['best','tie-first','tie-second'])
            self.assertGreater(cost['client_response_received_bytes'],0)
            session = make_local_query(self.clients['none'],vector(1),'text')
            result,_ = engine.search(self.clients['none'],session)
            self.assertEqual(result['ids'],[])
            session = make_local_query(self.client,vector(1),'text')
            result,_ = engine.search(self.client,session)
            self.assertEqual(result['ids'],['image'])
            with self.assertRaisesRegex(ValueError,'Replayed'):
                engine.search(self.client,session)

    def test_local_baseline_authenticates_full_collection(self):
        server = copy.deepcopy(self.engine.server_data)
        server['records'] = [r for r in server['records'] if r['id']!='a']
        with AuthorizedLocalEngine(server) as engine:
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                engine.search(self.client,make_local_query(self.client,vector(1),'image',3))


if __name__ == '__main__':
    unittest.main()
