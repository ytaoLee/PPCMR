import copy
import unittest

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ppcmr.attacks import attack_cases
from ppcmr.crypto import decrypt_content, sign
from ppcmr.dataset import synthetic_records
from ppcmr.protocol import (ClientContext, Owner, ProtocolError, aad_for,
                            make_request, verify_response)


def item(identifier, scope, modality, code):
    return {"id": identifier, "scope": scope, "modality": modality,
            "hash_code": code, "content": f"payload-{identifier}"}


UNSET = object()


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        # Manually known scores for an all-ones query: 8, 7, 6, 5, 4.
        self.records = [item("a", "blocked", "text", "ff"),
                        item("b", "allowed", "text", "fe"),
                        item("c", "blocked", "text", "fc"),
                        item("d", "allowed", "text", "f8"),
                        item("e", "allowed", "text", "f0"),
                        item("q", "allowed", "image", "ff")]
        self.owner = Owner("tests")
        self.grants = {"alice": ["allowed"], "bob": ["allowed", "blocked"], "none": []}
        self.server, self.clients = self.owner.publish(self.records, self.grants, bits=8)
        self.client = self.clients["alice"]
        self.request = make_request(self.client, "ff", "image", 3)
        self.response = self.server.search(self.request, self.client.credential)

    def verify(self, response=UNSET, request=None, context=None, decrypt=False):
        return verify_response(self.client if context is None else context,
                               self.request if request is None else request,
                               self.response if response is UNSET else response, decrypt)

    def test_authorize_before_ranking_with_known_expected_results(self):
        result = self.verify(decrypt=True)
        self.assertTrue(result.accepted)
        self.assertEqual(result.verified_ids, ("b", "d", "e"))
        self.assertEqual([r["score"] for r in self.response["results"]], [7, 5, 4])
        self.assertEqual(result.decrypted_contents, ("payload-b", "payload-d", "payload-e"))

    def test_naive_global_topk_then_filter_is_rejected(self):
        bad = copy.deepcopy(self.response)
        bad["results"] = [bad["results"][0]]  # Global a,b,c filtered to b.
        self.assertEqual(self.verify(bad).code, "RESULT_COUNT_MISMATCH")

    def test_authentic_lower_record_replaces_best_with_valid_full_witness(self):
        request = make_request(self.client, "ff", "image", 2)
        honest = self.server.search(request, self.client.credential)
        bad = copy.deepcopy(honest)
        bad["results"] = [{"id": "d", "score": 5, "content": self.server.contents["d"]},
                          {"id": "e", "score": 4, "content": self.server.contents["e"]}]
        # Full witnesses and ciphertexts are unmodified, so membership and integrity hold.
        self.assertEqual(bad["witnesses"], honest["witnesses"])
        self.assertEqual(self.verify(bad, request).code, "TOPK_MISMATCH")

    def test_zero_rights_is_an_authenticated_empty_answer(self):
        client = self.clients["none"]
        request = make_request(client, "ff", "image", 10)
        response = self.server.search(request, client.credential)
        result = self.verify(response, request, client)
        self.assertTrue(result.accepted)
        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(response["results"], [])

    def test_authorized_but_empty_target_partition(self):
        request = make_request(self.clients["bob"], "ff", "text", 10)
        response = self.server.search(request, self.clients["bob"].credential)
        result = self.verify(response, request, self.clients["bob"])
        self.assertTrue(result.accepted)
        self.assertEqual(result.verified_ids, ("q",))
        self.assertEqual(len(response["witnesses"]), 2)

    def test_k_larger_than_candidate_count(self):
        request = make_request(self.client, "ff", "image", 100)
        response = self.server.search(request, self.client.credential)
        self.assertEqual(self.verify(response, request).verified_ids, ("b", "d", "e"))

    def test_reverse_direction(self):
        request = make_request(self.client, "f0", "text", 2)
        response = self.server.search(request, self.client.credential)
        self.assertEqual(self.verify(response, request).verified_ids, ("q",))

    def test_tie_breaking_uses_id_ascending(self):
        records = [item("z", "s", "text", "f0"), item("a", "s", "text", "f0")]
        server, clients = self.owner.publish(records, {"alice": ["s"]}, bits=8)
        request = make_request(clients["alice"], "ff", "image", 1)
        response = server.search(request, clients["alice"].credential)
        self.assertEqual(verify_response(clients["alice"], request, response).verified_ids, ("a",))
        bad = copy.deepcopy(response)
        bad["results"] = [{"id": "z", "score": 4, "content": server.contents["z"]}]
        self.assertEqual(verify_response(clients["alice"], request, bad).code, "TOPK_MISMATCH")

    def test_all_generated_attack_scenarios_are_rejected(self):
        server, clients = self.owner.publish(synthetic_records(100, scopes=5),
                                             {"alice": ["project-00", "project-01"]})
        request = make_request(clients["alice"], "0" * 16, "image", 5)
        response = server.search(request, clients["alice"].credential)
        cases = attack_cases(server, clients["alice"], request, response)
        self.assertEqual(len(cases), 15)
        for case in cases:
            with self.subTest(attack=case["name"]):
                self.assertFalse(verify_response(clients["alice"], case["request"], case["response"]).accepted)

    def test_changed_score(self):
        bad = copy.deepcopy(self.response)
        bad["results"][0]["score"] = 0
        self.assertEqual(self.verify(bad).code, "TOPK_MISMATCH")

    def test_wrong_score_type_bool_is_rejected(self):
        bad = copy.deepcopy(self.response)
        bad["results"][0]["score"] = True
        self.assertEqual(self.verify(bad).code, "BAD_RESULT")

    def test_unauthorized_record_is_rejected(self):
        bad = copy.deepcopy(self.response)
        bad["results"][0] = {"id": "a", "score": 8, "content": self.server.contents["a"]}
        self.assertEqual(self.verify(bad).code, "UNAUTHORIZED_RESULT")

    def test_candidate_omission_preserving_count_is_rejected(self):
        bad = copy.deepcopy(self.response)
        bad["witnesses"][0]["records"][0]["id"] = "fake"
        self.assertEqual(self.verify(bad).code, "CANDIDATE_ROOT_MISMATCH")

    def test_duplicate_candidate_cannot_pad_an_omission(self):
        bad = copy.deepcopy(self.response)
        bad["witnesses"][0]["records"][0] = copy.deepcopy(bad["witnesses"][0]["records"][1])
        self.assertEqual(self.verify(bad).code, "DUPLICATE_RECORD")

    def test_reorder_witness_records_does_not_change_commitment(self):
        bad = copy.deepcopy(self.response)
        bad["witnesses"][0]["records"].reverse()
        self.assertTrue(self.verify(bad).accepted)

    def test_nonverifiable_baseline_output_is_not_accepted_as_proof(self):
        response = self.server.search(self.request, self.client.credential, include_witnesses=False)
        self.assertEqual(self.verify(response).code, "MISSING_PARTITION")

    def test_wrong_owner_key_rejects_self_signed_forged_state(self):
        attacker = Owner("tests")
        forged_manifest = sign(attacker.private_key, self.client.trusted_manifest["payload"], "ppcmr.snapshot.v1")
        context = ClientContext(self.client.public_key, forged_manifest, self.client.credential, {})
        self.assertEqual(self.verify(context=context).code, "BAD_MANIFEST_SIGNATURE")

    def test_forged_grant_rejected(self):
        credential = copy.deepcopy(self.client.credential)
        credential["payload"]["allowed_scopes"].append("blocked")
        context = ClientContext(self.client.public_key, self.client.trusted_manifest, credential, {})
        self.assertEqual(self.verify(context=context).code, "BAD_CREDENTIAL_SIGNATURE")

    def test_old_grant_rejected_after_revocation(self):
        server, clients = self.owner.publish(self.records, {"alice": []}, bits=8, epoch=2, permission_epoch=2)
        with self.assertRaises(ProtocolError) as caught:
            server.search(self.request, self.client.credential)
        self.assertEqual(caught.exception.code, "STALE_CREDENTIAL")
        new_request = make_request(clients["alice"], "ff", "image", 3)
        self.assertEqual(verify_response(clients["alice"], new_request, self.response).code, "STALE_RESPONSE")

    def test_stale_client_can_still_accept_old_snapshot_without_trusted_update(self):
        # Captures the explicit freshness limitation, rather than hiding it.
        self.owner.publish(self.records, {"alice": []}, bits=8, epoch=2, permission_epoch=2)
        self.assertTrue(self.verify().accepted)

    def test_response_replay_with_new_nonce(self):
        request = make_request(self.client, "ff", "image", 3)
        self.assertEqual(self.verify(request=request).code, "QUERY_BINDING_FAILED")

    def test_copied_credential_cannot_sign_another_query(self):
        request = copy.deepcopy(self.request)
        request["query_code"] = "00"
        with self.assertRaises(ProtocolError) as caught:
            self.server.search(request, self.client.credential)
        self.assertEqual(caught.exception.code, "BAD_REQUEST_SIGNATURE")

    def test_signed_query_from_different_client_rejected(self):
        request = make_request(self.clients["bob"], "ff", "image", 3)
        with self.assertRaises(ProtocolError) as caught:
            self.server.search(request, self.client.credential)
        self.assertEqual(caught.exception.code, "WRONG_SUBJECT")

    def test_same_modality_request_rejected(self):
        request = make_request(self.client, "ff", "image", 3)
        payload = {k: v for k, v in request.items() if k != "signature"}
        payload["target_modality"] = "image"
        request = {**payload, "signature": sign(self.client.query_signing_key, payload, "ppcmr.request.v1")["signature"]}
        with self.assertRaises(ProtocolError) as caught:
            self.server.search(request, self.client.credential)
        self.assertEqual(caught.exception.code, "BAD_REQUEST")

    def test_k_must_be_positive_integer(self):
        for k in (0, -1, True, 1.5):
            with self.subTest(k=k):
                request = make_request(self.client, "ff", "image", k)
                with self.assertRaises(ProtocolError):
                    self.server.search(request, self.client.credential)

    def test_content_commitment_binds_ciphertext(self):
        bad = copy.deepcopy(self.response)
        bad["results"][0]["content"] = self.server.contents["d"]
        self.assertEqual(self.verify(bad).code, "CONTENT_DIGEST_MISMATCH")

    def test_cloud_without_scope_key_cannot_decrypt_content(self):
        m = next(m for m in self.server.metadata if m["id"] == "b")
        with self.assertRaises(InvalidTag):
            decrypt_content(AESGCM.generate_key(bit_length=256), self.server.contents["b"],
                            aad_for(m, self.server.manifest["payload"]))

    def test_no_private_keys_in_public_protocol_transcript(self):
        self.assertNotIn("content_keys", self.response)
        self.assertNotIn("query_signing_key", self.request)
        self.assertFalse(hasattr(self.server, "content_keys"))

    def test_malformed_responses_fail_closed(self):
        malformed = [None, [], {}, {**self.response, "results": None},
                     {**self.response, "witnesses": [{"scope": [], "modality": "text", "records": []}]},
                     {**self.response, "results": [None]}, {**self.response, "owner_public_key": "attacker"}]
        for response in malformed:
            with self.subTest(response_type=type(response).__name__):
                self.assertFalse(self.verify(response=response).accepted)

    def test_duplicate_input_ids_rejected(self):
        with self.assertRaises(ProtocolError):
            self.owner.publish(self.records + [self.records[0]], self.grants, bits=8)

    def test_invalid_feature_width_rejected(self):
        records = copy.deepcopy(self.records)
        records[0]["hash_code"] = "ffff"
        with self.assertRaises(ProtocolError):
            self.owner.publish(records, self.grants, bits=8)

    def test_unknown_grant_scope_rejected(self):
        with self.assertRaises(ProtocolError):
            self.owner.publish(self.records, {"alice": ["unknown"]}, bits=8)

    def test_empty_database_with_empty_grant(self):
        server, clients = self.owner.publish([], {"alice": []}, bits=8)
        request = make_request(clients["alice"], "00", "image", 3)
        response = server.search(request, clients["alice"].credential)
        self.assertTrue(verify_response(clients["alice"], request, response).accepted)


if __name__ == "__main__":
    unittest.main()
