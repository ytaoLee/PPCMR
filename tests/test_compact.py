import copy
import unittest

import numpy as np

from ppcmr.compact_verify import (PRIME, SCORE_BOUND, coefficient,
                                 prepare_verifier_state, verify_compact)
from ppcmr.crypto import digest
from ppcmr.float_mpc import TwoCloudEngine, make_query, publish, verify_result
from test_float_mpc import vector


class CompactVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records = [
            {'id':'a','scope':'s','modality':'text','embedding':vector(.75),'content':'a'},
            {'id':'b','scope':'s','modality':'text','embedding':vector(.5),'content':'b'},
            {'id':'c','scope':'s','modality':'text','embedding':vector(.25),'content':'c'},
            {'id':'d','scope':'s','modality':'text','embedding':vector(-.5),'content':'d'},
            {'id':'zero','scope':'s','modality':'text','embedding':vector(0,1),'content':'zero'},
            {'id':'image','scope':'s','modality':'image','embedding':vector(1),'content':'image'},
            {'id':'blocked','scope':'t','modality':'text','embedding':vector(1),'content':'blocked'},
        ]
        server,shares,cls.clients = publish(records,{'alice':['s'],'none':[]},{'test':True},compact=True)
        cls.engine = TwoCloudEngine(server,shares)
        cls.client = cls.clients['alice']
        cls.state = prepare_verifier_state(cls.client)
        cls.session = make_query(cls.client,vector(1),'image',3)
        cls.result,cls.cost = cls.engine.search(cls.client,cls.session,witness_mode='compact')

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_honest_full_and_compact_agree_without_full_feature_witness(self):
        verdict = verify_compact(self.client,self.session,self.result,self.state)
        self.assertTrue(verdict['accepted'],verdict)
        self.assertEqual(verdict['ids'],['a','b','c'])
        self.assertTrue(all('feature' not in m for m in self.result['witnesses']))
        session = make_query(self.client,vector(1),'image',3)
        full,_ = self.engine.search(self.client,session)
        self.assertTrue(verify_result(self.client,session,full)['accepted'])
        self.assertEqual(full['scores'],self.result['scores'])
        self.assertEqual(full['results'],self.result['results'])

    def test_zero_rights_reverse_negative_and_subnormal_queries(self):
        tiny = np.nextafter(np.float32(0),np.float32(1))
        for client,feature,modality in ((self.clients['none'],vector(1),'image'),
                (self.client,vector(-.5),'text'),(self.client,vector(tiny,-tiny),'image')):
            session = make_query(client,feature,modality,10)
            compact,_ = self.engine.search(client,session,'compact')
            verdict = verify_compact(client,session,compact)
            self.assertTrue(verdict['accepted'],verdict)

    def test_score_modulus_and_ring_aliases_are_rejected(self):
        for delta in (1,PRIME,1<<320):
            bad = copy.deepcopy(self.result)
            bad['scores'][0] += delta
            self.assertFalse(verify_compact(self.client,self.session,bad,self.state)['accepted'])

    def test_coverage_ranking_and_content_mutations_are_rejected(self):
        for kind in ('coverage','scope','order','content','duplicate','format'):
            bad = copy.deepcopy(self.result)
            if kind=='coverage':
                bad['witnesses'].pop(); bad['scores'].pop()
            if kind=='scope': bad['witnesses'][0]['scope']='t'
            if kind=='order': bad['results'].reverse()
            if kind=='content': bad['results'][0]['content']['ciphertext']='AAAA'
            if kind=='duplicate': bad['witnesses'].append(copy.deepcopy(bad['witnesses'][0]))
            if kind=='format': bad['proof_format']='unknown'
            with self.subTest(kind=kind):
                self.assertFalse(verify_compact(self.client,self.session,bad,self.state)['accepted'])

    def test_private_blob_and_stale_cache_are_rejected(self):
        client = copy.copy(self.client)
        client.verification_sketches = copy.deepcopy(client.verification_sketches)
        client.verification_sketches[('s','text')]['ciphertext']='AAAA'
        with self.assertRaisesRegex(ValueError,'ciphertext'):
            prepare_verifier_state(client)
        state = copy.deepcopy(self.state)
        state['snapshot_digest']='0'*64
        self.assertFalse(verify_compact(self.client,self.session,self.result,state)['accepted'])
        self.assertNotIn('seed',str(self.engine.server_data))
        self.assertNotIn('columns',str(self.engine.server_data))

    def test_ablation_shows_necessity_of_coverage_and_score_checks(self):
        bad = copy.deepcopy(self.result)
        i = next(i for i,m in enumerate(bad['witnesses']) if m['id']=='zero')
        bad['witnesses'].pop(i); bad['scores'].pop(i)
        self.assertFalse(verify_compact(self.client,self.session,bad,self.state)['accepted'])
        self.assertTrue(verify_compact(self.client,self.session,bad,self.state,check_coverage=False)['accepted'])
        bad = copy.deepcopy(self.result)
        bad['scores'][0]+=1
        self.assertFalse(verify_compact(self.client,self.session,bad,self.state)['accepted'])
        self.assertTrue(verify_compact(self.client,self.session,bad,self.state,check_scores=False)['accepted'])

    def test_seed_disclosure_enables_bounded_false_score_forgery(self):
        # Demonstrates why the private state cannot be published or given to
        # colluding servers. Euclidean reduction finds a small modular kernel.
        private = self.state['partitions'][('s','text')]
        weights = [coefficient(private['seed'],private['context'],m['id'])
                   for m in self.result['witnesses'][:2]]
        k = weights[1]*pow(weights[0],-1,PRIME)%PRIME
        r0,r1,t0,t1 = PRIME,k,0,1
        while r1 > 1<<261:
            quotient = r0//r1
            r0,r1 = r1,r0-quotient*r1
            t0,t1 = t1,t0-quotient*t1
        delta0,delta1 = r1,-t1
        self.assertEqual((weights[0]*delta0+weights[1]*delta1)%PRIME,0)
        self.assertLess(max(abs(delta0),abs(delta1)),SCORE_BOUND//4)
        bad = copy.deepcopy(self.result)
        bad['scores'][0]+=delta0; bad['scores'][1]+=delta1
        self.assertNotEqual(bad['scores'],self.result['scores'])
        self.assertTrue(verify_compact(self.client,self.session,bad,self.state)['accepted'])

    def test_compact_epoch_change_rejects_old_credential_and_private_state(self):
        from ppcmr.crypto import new_owner_key
        owner = new_owner_key()
        records = [{'id':'a','scope':'s','modality':'text','embedding':vector(1),'content':'x'}]
        _,_,old = publish(records,{'u':['s']},{},epoch=1,owner_key=owner,compact=True)
        old_state = prepare_verifier_state(old['u'])
        server,shares,new = publish(records,{'u':[]},{},epoch=2,owner_key=owner,compact=True)
        with TwoCloudEngine(server,shares) as engine:
            with self.assertRaisesRegex(ValueError,'Stale'):
                engine.search(old['u'],make_query(old['u'],vector(1),'image'),'compact')
            session = make_query(new['u'],vector(1),'image')
            result,_ = engine.search(new['u'],session,'compact')
            self.assertTrue(verify_compact(new['u'],session,result)['accepted'])
            self.assertFalse(verify_compact(new['u'],session,result,old_state)['accepted'])


if __name__=='__main__':
    unittest.main()
