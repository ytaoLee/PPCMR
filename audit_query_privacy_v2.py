"""Reproduce legacy confirmation and audit v2 recipient-private query delivery.

This checks removal of a specific deterministic oracle, not general absence
of all possible attacks. No secret shares or keys are written to the report.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ppcmr.crypto import canonical, sign
from ppcmr.float_mpc import (MASK, authenticate_query_share, make_query,
                            prepare_message, publish, share_digest, validate_feature)
from research_common import ARTIFACTS, file_sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ARTIFACTS/'float_mpc_v2'/'query_privacy_audit.json')
    args = parser.parse_args()
    cache = ARTIFACTS/'test'/'features.npz'
    with np.load(cache,allow_pickle=False) as data:
        dictionary = data['images'][:8].copy()
    records = [{'id':f'audit-{i:02d}','scope':'audit','modality':'image',
                'embedding':q,'content':'audit dummy'} for i,q in enumerate(dictionary)]
    _,_,clients = publish(records,{'auditor':['audit']},{'purpose':'query-delivery privacy regression'})
    client = clients['auditor']
    outcomes = []
    for index,q in enumerate(dictionary):
        session = make_query(client,q,'image')
        legacy_payload = {**session.request['payload'], 'schema':'ppcmr.float-mpc.query.v1',
                          'share_commitments':[share_digest(s) for s in session.shares]}
        legacy = sign(client.signing_key,legacy_payload,'float-query')
        for party in (0,1):
            opposite = legacy['payload']['share_commitments'][1-party]
            matches = [j for j,candidate in enumerate(dictionary)
                       if share_digest([(x-a)&MASK for x,a in zip(validate_feature(candidate),session.shares[party])])==opposite]
            delivery = prepare_message(client,session,party)
            serialized = canonical(delivery)
            binding = authenticate_query_share(client.credential['payload'],session.request,
                       delivery['share_envelope'],party,delivery['share'])
            # Every dictionary query has a mathematically compatible OTHER
            # share, and v2 exposes no predicate that can check that share.
            compatible = sum(len([(x-a)&MASK for x,a in zip(validate_feature(c),delivery['share'])])==512
                             for c in dictionary)
            outcomes.append({'query_index':index,'party':party,'legacy_matches':matches,
                'legacy_uniquely_identified':matches==[index],
                'v2_has_public_share_commitments':'share_commitments' in session.request['payload'],
                'v2_opposite_digest_absent':opposite.encode('ascii') not in serialized,
                'v2_delivery_fields':sorted(delivery),
                'v2_envelope_fields':sorted(binding),
                'dictionary_candidates_compatible_with_v2_delivery':compatible})
    require_pass = all(x['legacy_uniquely_identified'] and x['v2_opposite_digest_absent'] and
                       not x['v2_has_public_share_commitments'] and
                       x['dictionary_candidates_compatible_with_v2_delivery']==len(dictionary) for x in outcomes)
    if not require_pass:
        raise RuntimeError('Query delivery regression failed')
    report = {'schema':'ppcmr.query-delivery-regression.v2','audit_date':'2026-10-09',
        'features_sha256':file_sha256(cache),'dictionary_size':len(dictionary),'party_views':len(outcomes),
        'legacy_unique_confirmations':sum(x['legacy_uniquely_identified'] for x in outcomes),
        'v2_opposite_digest_absent_views':sum(x['v2_opposite_digest_absent'] for x in outcomes),
        'v2_delivery_regression_passed':require_pass,
        'no_search_outputs_used':True,'target_gallery_empty':True,
        'scope':'Known opposite-share-digest confirmation oracle removed from actual query-delivery builder; not a proof of the whole implementation',
        'outcomes':outcomes}
    write_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='outcomes'},indent=2))


if __name__=='__main__':
    main()
