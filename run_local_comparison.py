"""Paired v2 MPC/full verifier vs authorized-download/local exact search.

Same published data, signed permissions, exact dot products, tie rule, and
original JPEG/text contents. Fully counted online totals include local query
construction, dispatch, preprocessing (MPC), response serialization and final
content verification; exclude initial publication, encoder extraction and
worker launch. Each query uses fresh triples/nonces and no feature caching.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import argparse
import importlib.metadata
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np

from float_reference import encode_fp32, exact_dot_encoded
from ppcmr.float_mpc import TwoCloudEngine, make_query, publish, verify_result
from ppcmr.local_reference import AuthorizedLocalEngine, make_local_query
from research_common import ARTIFACTS, entity_arrays, file_sha256, load_features, write_json
from run_float_mpc_research import build_records


def perform(method, engine, client, feature, modality, reference):
    start = perf_counter()
    if method == 'mpc_full_verify':
        session = make_query(client,feature,modality,10)
        query_seconds = perf_counter()-start
        result,cost = engine.search(client,session)
        verified_start = perf_counter()
        verdict = verify_result(client,session,result)
        verify_seconds = perf_counter()-verified_start
        scores = result['scores']
    else:
        session = make_local_query(client,feature,modality,10)
        query_seconds = perf_counter()-start
        verdict,cost = engine.search(client,session)
        verify_seconds = 0.0  # Authentication, local ranking and content checks are inside search.
        scores = verdict['scores']
    total = perf_counter()-start
    if not verdict['accepted'] or scores != reference['scores'] or verdict['ids'] != reference['ids']:
        raise RuntimeError('Paired baseline differs from independent original-cache exact reference')
    # Compare actual decrypted content to the original records, outside timing.
    if verdict['contents'] != reference['contents']:
        raise RuntimeError('Returned content differs from original source')
    return {'method':method,'query_construction_seconds':query_seconds,
        'final_verification_seconds':verify_seconds,'online_total_seconds':total,
        'exact_scores_equal':True,'exact_order_equal':True,'original_contents_equal':True,**cost}


def summarize(rows, sizes):
    result = []
    for size in sizes:
        for direction in ('image-to-text','text-to-image'):
            mpc = [r for r in rows if r['images']==size and r['direction']==direction and r['method']=='mpc_full_verify']
            local = [r for r in rows if r['images']==size and r['direction']==direction and r['method']=='authorized_local']
            pairs = list(zip(mpc,local))
            ratios = [a['online_total_seconds']/b['online_total_seconds'] for a,b in pairs]
            def detail(group):
                times = [r['online_total_seconds'] for r in group]
                return {'timed_runs':len(group),'median_total_ms':statistics.median(times)*1000,
                    'min_total_ms':min(times)*1000,'max_total_ms':max(times)*1000,
                    'median_client_payload_MB':statistics.median(r['client_total_transport_payload_bytes'] for r in group)/1e6,
                    'median_peer_payload_MB':statistics.median(r.get('peer_opening_sent_bytes',0) for r in group)/1e6,
                    'median_dealer_payload_MB':statistics.median(r.get('trusted_preprocessing_sent_bytes',0) for r in group)/1e6,
                    'median_result_JSON_MB':statistics.median(r['result_json_bytes'] for r in group)/1e6}
            result.append({'images':size,'direction':direction,'candidate_count':mpc[0]['candidate_count'],
                'mpc':detail(mpc),'local':detail(local),'median_paired_time_ratio_mpc_over_local':statistics.median(ratios),
                'median_paired_extra_total_ms':statistics.median((a['online_total_seconds']-b['online_total_seconds'])*1000 for a,b in pairs)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=ARTIFACTS/'test')
    parser.add_argument('--images',type=int,nargs='+',default=[100,250,1000])
    parser.add_argument('--queries',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output',type=Path,default=ARTIFACTS/'float_mpc_v2'/'local_comparison.json')
    args = parser.parse_args()
    manifest,provenance,images,texts = load_features(args.folder)
    _,_,_,_,scopes,_ = entity_arrays(manifest)
    rows, deployment = [],[]
    for size in args.images:
        if not 10 <= size <= len(images) or args.queries<1 or args.repeats<1:
            raise ValueError('Invalid benchmark parameters')
        records = build_records(manifest,images,texts,size)
        cohort = np.flatnonzero(scopes[:size]=='scope-00')[:args.queries]
        if len(cohort)!=args.queries:
            raise ValueError('Not enough fixed query pairs')
        original = sorted(records,key=lambda r:r['id'])
        references = {}
        for direction in ('image-to-text','text-to-image'):
            candidates = [r for r in original if r['modality']==('text' if direction=='image-to-text' else 'image')]
            for i in cohort:
                feature = images[i] if direction=='image-to-text' else texts[5*i]
                q = encode_fp32(feature)
                scores = [exact_dot_encoded(q,encode_fp32(r['embedding'])) for r in candidates]
                order = sorted(range(len(candidates)),key=lambda j:(-scores[j],candidates[j]['id']))[:10]
                references[(direction,int(i))] = {'scores':scores,'ids':[candidates[j]['id'] for j in order],
                    'contents':[candidates[j]['content'] for j in order]}
        start = perf_counter()
        server,shares,clients = publish(records,{'all':[f'scope-{i:02d}' for i in range(10)]},
            {'features_sha256':provenance['features_sha256'],'manifest_sha256':provenance['manifest_sha256'],
             'subset_images':size,'comparison':'equal authorized exact retrieval'})
        publish_seconds = perf_counter()-start
        client = clients['all']
        with TwoCloudEngine(server,shares) as mpc, AuthorizedLocalEngine(server) as local:
            engines = {'mpc_full_verify':mpc,'authorized_local':local}
            deployment.append({'images':size,'records':len(records),'publication_seconds':publish_seconds,
                'mpc_worker_launch_seconds':mpc.startup_seconds,'local_worker_launch_seconds':local.startup_seconds})
            for direction in ('image-to-text','text-to-image'):
                modality = 'image' if direction=='image-to-text' else 'text'
                for method in engines:
                    i = int(cohort[0])
                    feature = images[i] if modality=='image' else texts[5*i]
                    perform(method,engines[method],client,feature,modality,references[(direction,i)])
                for query_number,i0 in enumerate(cohort):
                    i = int(i0)
                    feature = images[i] if modality=='image' else texts[5*i]
                    for repeat in range(args.repeats):
                        order = list(engines) if (query_number+repeat)%2==0 else list(reversed(engines))
                        for method in order:
                            row = perform(method,engines[method],client,feature,modality,references[(direction,i)])
                            rows.append({'images':size,'direction':direction,'query_number':query_number,
                                'query_cache_image_index':i,'repeat':repeat,'method_order':order,**row})
                    print(size,direction,'query',query_number,'paired repeats',args.repeats,'passed',flush=True)
        # Preserve a useful partial artifact after each completed scale.
        report = {'schema':'ppcmr.equal-authorized-local-comparison.v1','date':'2026-10-09',
            'features_sha256':provenance['features_sha256'],'manifest_sha256':provenance['manifest_sha256'],
            'runtime':{'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,
                'cryptography':importlib.metadata.version('cryptography'),'BLAS_threads':4},
            'source_sha256':{str(p.relative_to(Path(__file__).parent)):file_sha256(p) for p in
                (Path(__file__),Path(__file__).parent/'ppcmr/float_mpc.py',Path(__file__).parent/'ppcmr/local_reference.py')},
            'all_permissions':True,'distinct_query_pairs':args.queries,'repeats_per_query_per_method':args.repeats,
            'warmup_runs_per_method_per_scale':2,'fresh_triples_and_nonces':True,'feature_cache_across_queries':False,
            'timing_includes':['local request construction','authenticated feature delivery','MPC triple generation and delivery when used',
                'score computation and exact ranking','assembled response canonical serialization','full final content verification'],
            'timing_excludes':['initial feature extraction','database publication','worker launch',
                'independent original-cache reference','outside-timing original-content equality check'],
            'payload_units':'bytes / decimal MB; serialized IPC payloads, excluding framing, no WAN measurement',
            'scope':'Same-host CPU experiment, full authorization, repeated timings of three fixed query pairs; not population recall or remote deployment',
            'deployment':deployment,'measurements':rows,'summary':summarize(rows,[d['images'] for d in deployment])}
        write_json(args.output,report)
        print('completed scale',size,flush=True)
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    main()
