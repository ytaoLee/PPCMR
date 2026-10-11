"""Regression checks for public data-confirmation and state-length leaks."""
import copy
import hashlib
import json
import unittest

from ppcmr.compact_verify import prepare_verifier_state, sketch_context
from ppcmr.crypto import (decrypt_content, digest, encrypt_content, new_owner_key,
                         sign)
from ppcmr.float_mpc import TwoCloudEngine, authenticate, make_query, publish
from ppcmr.publication_policy import (FEATURE_PROFILE, POLICY,
                                      SKETCH_PADDED_BYTES)
from test_float_mpc import vector


class PublicationPrivacyTests(unittest.TestCase):
    def records(self, value=1):
        return [{'id':'a','scope':'s','modality':'text',
                 'embedding':vector(value),'content':'same length'}]

    def test_private_provenance_never_enters_server_publication(self):
        # Known-cache digests previously confirmed a private dataset directly.
        hashes = [hashlib.sha256(self.records(v)[0]['embedding'].tobytes()).hexdigest()
                  for v in (0,1)]
        for value, cache_hash in zip((0,1), hashes):
            server,_,_ = publish(self.records(value),{'u':['s']},
                    {'features_sha256':cache_hash,'nested':{'private':cache_hash}},compact=True)
            public = json.dumps(server,sort_keys=True)
            self.assertNotIn(cache_hash,public)
            self.assertNotIn('features_sha256',public)
            snapshot = server['snapshot']['payload']
            self.assertNotIn('provenance',snapshot)
            self.assertEqual(snapshot['publication_policy'],POLICY)
            self.assertEqual(snapshot['public_feature_profile'],FEATURE_PROFILE)

    def test_all_partition_sketches_have_fixed_encrypted_length(self):
        sizes = []
        for value in (0,1,-1):
            _,_,clients = publish(self.records(value),{'u':['s']},{},compact=True)
            client = clients['u']
            state = prepare_verifier_state(client)
            self.assertEqual(len(state['partitions']),2)  # includes empty image partition
            for key, blob in client.verification_sketches.items():
                plaintext = decrypt_content(client.scope_keys[key[0]],blob,
                                             sketch_context(client.snapshot['payload'],*key))
                self.assertEqual(len(plaintext.encode('utf-8')),SKETCH_PADDED_BYTES)
                sizes.append(len(blob['ciphertext']))
        self.assertEqual(len(set(sizes)),1)

    def test_authenticated_legacy_policy_is_rejected_before_workers_start(self):
        owner = new_owner_key()
        server,shares,clients = publish(self.records(),{'u':['s']},{},owner_key=owner,compact=True)
        client = copy.copy(clients['u'])
        payload = copy.deepcopy(server['snapshot']['payload'])
        payload['schema'] = 'ppcmr.float-mpc.snapshot.v1'
        payload.pop('publication_policy')
        payload.pop('public_feature_profile')
        payload.pop('private_sketch_plaintext_bytes')
        payload['provenance'] = {'features_sha256':'0'*64}
        legacy = sign(owner,payload,'float-snapshot')
        server = {**server,'snapshot':legacy}
        client.snapshot = legacy
        credential = {**client.credential['payload'],'snapshot_digest':digest(legacy)}
        client.credential = sign(owner,credential,'float-credential')
        with self.assertRaisesRegex(ValueError,'publication policy'):
            TwoCloudEngine(server,shares)
        with self.assertRaisesRegex(ValueError,'publication policy'):
            authenticate(client.owner_public,legacy,client.credential,
                         make_query(client,vector(1),'image').request)
        with self.assertRaisesRegex(ValueError,'publication policy'):
            prepare_verifier_state(client)

    def test_authentic_but_unpadded_private_state_is_rejected(self):
        # Honest-owner signing isolates the padding check from hash/signature checks.
        owner = new_owner_key()
        _,_,clients = publish(self.records(),{'u':['s']},{},owner_key=owner,compact=True)
        client = clients['u']
        key = ('s','text')
        context = sketch_context(client.snapshot['payload'],*key)
        plaintext = decrypt_content(client.scope_keys['s'],client.verification_sketches[key],context)
        blob = encrypt_content(client.scope_keys['s'],plaintext.rstrip(),context)
        payload = copy.deepcopy(client.snapshot['payload'])
        part = next(p for p in payload['partitions'] if (p['scope'],p['modality'])==key)
        part['sketch_digest'] = digest(blob)
        client.snapshot = sign(owner,payload,'float-snapshot')
        client.credential = sign(owner,{**client.credential['payload'],
                                     'snapshot_digest':digest(client.snapshot)},'float-credential')
        client.verification_sketches[key] = blob
        with self.assertRaisesRegex(ValueError,'padding length'):
            prepare_verifier_state(client)


if __name__=='__main__':
    unittest.main()
