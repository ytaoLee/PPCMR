"""Dataset-free demonstration of exact, authorized, verifiable FP32 retrieval."""
import argparse
import copy
import json

import numpy as np

from float_reference import exact_dot
from ppcmr.cached_mpc import CachedMatrixEngine
from ppcmr.compact_verify import prepare_verifier_state, verify_compact
from ppcmr.float_mpc import TwoCloudEngine, make_query, publish


def vector(first, second=0):
    result = np.zeros(512, dtype=np.float32)
    result[:2] = [first, second]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['fresh', 'cached'], default='cached')
    args = parser.parse_args()
    records = [
        {'id': 'image-a', 'scope': 'allowed', 'modality': 'image',
         'embedding': vector(1), 'content': 'synthetic image A'},
        {'id': 'image-b', 'scope': 'allowed', 'modality': 'image',
         'embedding': vector(.5), 'content': 'synthetic image B'},
        {'id': 'text-a', 'scope': 'allowed', 'modality': 'text',
         'embedding': vector(.75), 'content': 'synthetic caption A'},
        {'id': 'text-b', 'scope': 'allowed', 'modality': 'text',
         'embedding': vector(.5), 'content': 'synthetic caption B'},
        {'id': 'text-denied', 'scope': 'denied', 'modality': 'text',
         'embedding': vector(1), 'content': 'unauthorized caption'},
    ]
    server, shares, clients = publish(records, {'alice': ['allowed']},
                                      {'demo': 'synthetic'}, compact=True)
    client = clients['alice']
    state = prepare_verifier_state(client)
    engine_type = TwoCloudEngine if args.engine == 'fresh' else CachedMatrixEngine
    summaries = []
    with engine_type(server, shares) as engine:
        for modality in ('image', 'text'):
            feature = vector(1)
            session = make_query(client, feature, modality, 2)
            response, cost = engine.search(client, session, 'compact')
            verified = verify_compact(client, session, response, state)
            assert verified['accepted'], verified
            candidates = sorted((r for r in records if r['scope'] == 'allowed'
                                 and r['modality'] != modality), key=lambda r: r['id'])
            expected = [exact_dot(feature, r['embedding']) for r in candidates]
            assert response['scores'] == expected
            order = sorted(range(len(candidates)), key=lambda i: (-expected[i], candidates[i]['id']))[:2]
            assert verified['ids'] == [candidates[i]['id'] for i in order]
            assert verified['contents'] == [candidates[i]['content'] for i in order]
            forged = copy.deepcopy(response)
            forged['scores'][0] += 1
            assert not verify_compact(client, session, forged, state)['accepted']
            omitted = copy.deepcopy(response)
            omitted['witnesses'].pop()
            omitted['scores'].pop()
            assert not verify_compact(client, session, omitted, state)['accepted']
            summaries.append({'query_modality': modality, 'ids': verified['ids'],
                              'exact_scores_equal': True, 'modified_score_rejected': True,
                              'missing_candidate_rejected': True,
                              'cloud_process_ids': cost['cloud_process_ids']})
    print(json.dumps({'engine': args.engine, 'queries': summaries}, indent=2))


if __name__ == '__main__':
    main()
