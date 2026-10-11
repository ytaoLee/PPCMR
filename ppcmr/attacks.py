"""Adversarial responses, including valid-record top-k substitution attacks."""

import copy

from .crypto import decode, encode
from .protocol import make_request, similarity, verify_response


def attack_cases(server, client, request, honest):
    cases = []

    def add(name, description, response, expected_request=None):
        cases.append({"name": name, "description": description,
                      "request": expected_request or request, "response": response})

    altered = copy.deepcopy(honest)
    altered["results"][0]["score"] += 1
    add("score_tampering", "Modify a returned similarity score", altered)

    rights = client.credential["payload"]["allowed_scopes"]
    authorized = [m for m in server.metadata if m["scope"] in rights
                  and m["modality"] == request["target_modality"]]
    returned_ids = {r["id"] for r in honest["results"]}
    remaining = [m for m in authorized if m["id"] not in returned_ids]
    if remaining:
        low = min(remaining, key=lambda m: similarity(request["query_code"], m["hash_code"],
                                                     server.manifest["payload"]["hash_bits"]))
        altered = copy.deepcopy(honest)
        # Metadata and ciphertext remain authentic. Membership checks alone pass.
        replacement = {"id": low["id"], "score": similarity(request["query_code"], low["hash_code"],
                        server.manifest["payload"]["hash_bits"]), "content": copy.deepcopy(server.contents[low["id"]])}
        altered["results"][0] = replacement
        altered["results"].sort(key=lambda r: (-r["score"], r["id"]))
        add("valid_low_score_substitution", "Replace the best result with an authentic lower-ranked authorized record", altered)

    if len(honest["results"]) > 1:
        altered = copy.deepcopy(honest)
        altered["results"][0], altered["results"][1] = altered["results"][1], altered["results"][0]
        add("ranking_tampering", "Swap the first two results", altered)
        altered = copy.deepcopy(honest)
        altered["results"][-1] = copy.deepcopy(altered["results"][0])
        add("duplicate_result", "Fill a result slot with a duplicate record", altered)

    altered = copy.deepcopy(honest)
    altered["results"].pop(0)
    add("result_omission", "Return fewer results than required", altered)

    outsiders = [m for m in server.metadata if m["scope"] not in rights
                 and m["modality"] == request["target_modality"]]
    if outsiders:
        outsider = outsiders[0]
        altered = copy.deepcopy(honest)
        altered["results"][0] = {"id": outsider["id"],
            "score": similarity(request["query_code"], outsider["hash_code"], server.manifest["payload"]["hash_bits"]),
            "content": copy.deepcopy(server.contents[outsider["id"]])}
        add("unauthorized_result", "Return an authentic result outside the grant", altered)
        altered = copy.deepcopy(honest)
        altered["witnesses"].append({"scope": outsider["scope"], "modality": outsider["modality"], "records": [outsider]})
        add("unauthorized_partition", "Add evidence from an unauthorized scope", altered)

    altered = copy.deepcopy(honest)
    nonempty = next(w for w in altered["witnesses"] if w["records"])
    nonempty["records"].pop(0)
    add("candidate_omission", "Remove an authorized candidate from the evidence", altered)

    altered = copy.deepcopy(honest)
    nonempty = next(w for w in altered["witnesses"] if w["records"])
    nonempty["records"][0]["id"] = "fabricated-candidate"
    add("same_size_candidate_substitution", "Keep the candidate count but replace committed metadata", altered)

    altered = copy.deepcopy(honest)
    nonempty = next(w for w in altered["witnesses"] if w["records"])
    record = nonempty["records"][0]
    bits = server.manifest["payload"]["hash_bits"]
    record["hash_code"] = f"{int(record['hash_code'], 16) ^ 1:0{bits // 4}x}"
    add("feature_tampering", "Change one candidate hash bit", altered)

    altered = copy.deepcopy(honest)
    altered["witnesses"].pop(0)
    add("authorized_partition_omission", "Omit a complete authorized scope", altered)

    altered = copy.deepcopy(honest)
    altered["witnesses"].append(copy.deepcopy(altered["witnesses"][0]))
    add("duplicate_partition", "Duplicate a partition witness", altered)

    altered = copy.deepcopy(honest)
    blob = altered["results"][0]["content"]
    ciphertext = bytearray(decode(blob["ciphertext"]))
    ciphertext[0] ^= 1
    blob["ciphertext"] = encode(ciphertext)
    add("ciphertext_tampering", "Change an encrypted content byte", altered)

    fresh_request = make_request(client, request["query_code"], request["query_modality"], request["k"])
    add("old_query_response_replay", "Reuse a response for the same query with a fresh nonce", copy.deepcopy(honest), fresh_request)

    different_code = f"{int(request['query_code'], 16) ^ 1:0{bits // 4}x}"
    different_request = make_request(client, different_code, request["query_modality"], request["k"])
    add("different_query_substitution", "Present another query's response", copy.deepcopy(honest), different_request)
    return cases


def evaluate_attacks(server, client, request, honest):
    outcomes = []
    for case in attack_cases(server, client, request, honest):
        verification = verify_response(client, case["request"], case["response"])
        outcomes.append({"attack": case["name"], "description": case["description"],
                         "detected": not verification.accepted, "reason_code": verification.code})
    return outcomes
