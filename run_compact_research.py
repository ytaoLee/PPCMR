"""Real COCO compact/private verifier vs full-feature reference, plus ablations."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import argparse
import copy
import importlib.metadata
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np

from float_reference import encode_fp32, exact_dot_encoded
from ppcmr.compact_verify import PRIME, prepare_verifier_state, verify_compact
from ppcmr.crypto import canonical
from ppcmr.float_mpc import TwoCloudEngine, make_query, publish, verify_result
from research_common import ARTIFACTS, entity_arrays, file_sha256, load_features, write_json
from run_float_mpc_research import build_records, tamper_cases


VARIANTS = {'complete':{},'no_catalog_coverage':{'check_coverage':False},
            'no_linear_score_check':{'check_scores':False},'no_order_check':{'check_ranking':False}}


def perform(engine,client,state,feature,modality,reference,mode):
    start = perf_counter()
    session = make_query(client,feature,modality,10)
    query_seconds = perf_counter()-start
    response,cost = engine.search(client,session,mode)
    verify_started = perf_counter()
    verdict = verify_compact(client,session,response,state) if mode=='compact' else verify_result(client,session,response)
    verification_seconds = perf_counter()-verify_started
    total = perf_counter()-start
    if not verdict['accepted'] or response['scores']!=reference['scores'] or verdict['ids']!=reference['ids']:
        raise RuntimeError('Compact/full run differs from independent exact reference')
    if verdict['contents']!=reference['contents']:
        raise RuntimeError('Returned plaintext differs from original content')
    return session,response,{'mode':mode,'query_construction_seconds':query_seconds,
        'verification_seconds':verification_seconds,'online_total_seconds':total,
        'exact_scores_equal':True,'exact_order_equal':True,'original_contents_equal':True,**cost}


def summarize(rows,sizes):
    summaries = []
    for size in sizes:
        for direction in ('image-to-text','text-to-image'):
            group = {mode:[r for r in rows if r['images']==size and r['direction']==direction and r['mode']==mode]
                     for mode in ('full','compact')}
            detail = lambda g:{'runs':len(g),
                'median_verify_ms':statistics.median(r['verification_seconds'] for r in g)*1000,
                'median_total_ms':statistics.median(r['online_total_seconds'] for r in g)*1000,
                'min_total_ms':min(r['online_total_seconds'] for r in g)*1000,
                'max_total_ms':max(r['online_total_seconds'] for r in g)*1000,
                'median_client_payload_MB':statistics.median(r['client_total_transport_payload_bytes'] for r in g)/1e6,
                'median_response_JSON_MB':statistics.median(r['result_json_bytes'] for r in g)/1e6,
                'median_peer_MB':statistics.median(r['peer_opening_sent_bytes'] for r in g)/1e6,
                'median_dealer_MB':statistics.median(r['trusted_preprocessing_sent_bytes'] for r in g)/1e6}
            pairs = list(zip(group['full'],group['compact']))
            summaries.append({'images':size,'direction':direction,'candidate_count':group['full'][0]['candidate_count'],
                'full':detail(group['full']),'compact':detail(group['compact']),
                'median_paired_verification_speedup':statistics.median(a['verification_seconds']/b['verification_seconds'] for a,b in pairs),
                'median_paired_total_speedup':statistics.median(a['online_total_seconds']/b['online_total_seconds'] for a,b in pairs)})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',type=int,nargs='+',default=[100,250,1000])
    parser.add_argument('--queries',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=ARTIFACTS/'float_compact')
    args = parser.parse_args()
    manifest,provenance,images,texts = load_features(ARTIFACTS/'test')
    *_,scopes,tscopes = entity_arrays(manifest)
    timed,functional,attacks,ablation,deployments = [],[],[],[],[]
    for size in args.images:
        if not 10<=size<=len(images) or args.queries<1 or args.repeats<1:
            raise ValueError('Invalid configuration')
        records = build_records(manifest,images,texts,size)
        cohort = np.flatnonzero(scopes[:size]=='scope-00')[:args.queries]
        if len(cohort)!=args.queries:
            raise ValueError('Not enough query pairs')
        grants = {'none':[],'one':['scope-00'],'half':[f'scope-{j:02d}' for j in range(5)],
                  'all':[f'scope-{j:02d}' for j in range(10)]}
        references = {}
        for subject,allowed in grants.items():
            for direction in ('image-to-text','text-to-image'):
                candidates = [r for r in sorted(records,key=lambda r:r['id']) if r['scope'] in allowed and
                              r['modality']==('text' if direction=='image-to-text' else 'image')]
                for i0 in cohort:
                    i = int(i0)
                    q = encode_fp32(images[i] if direction=='image-to-text' else texts[5*i])
                    scores = [exact_dot_encoded(q,encode_fp32(r['embedding'])) for r in candidates]
                    order = sorted(range(len(candidates)),key=lambda j:(-scores[j],candidates[j]['id']))[:10]
                    references[(subject,direction,i)] = {'scores':scores,'ids':[candidates[j]['id'] for j in order],
                        'contents':[candidates[j]['content'] for j in order]}
        start = perf_counter()
        server,shares,clients = publish(records,grants,{'features_sha256':provenance['features_sha256'],
            'manifest_sha256':provenance['manifest_sha256'],'subset_images':size},compact=True)
        publication_seconds = perf_counter()-start
        state,init_seconds,private_bytes = {},{},{}
        for subject,client in clients.items():
            start = perf_counter()
            state[subject] = prepare_verifier_state(client)
            init_seconds[subject] = perf_counter()-start
            private_bytes[subject] = len(canonical({f'{a}:{m}':blob for (a,m),blob in client.verification_sketches.items()}))
        with TwoCloudEngine(server,shares) as engine:
            deployments.append({'images':size,'records':len(records),'publication_seconds':publication_seconds,
                'owner_private_sketch_seconds':server['publication_metrics']['private_sketch_generation_seconds'],
                'private_state_initialization_seconds':init_seconds,'private_state_provisioned_JSON_bytes':private_bytes,
                'worker_launch_seconds':engine.startup_seconds})
            for direction in ('image-to-text','text-to-image'):
                modality = 'image' if direction=='image-to-text' else 'text'
                client = clients['all']
                i = int(cohort[0]); feature = images[i] if modality=='image' else texts[5*i]
                for mode in ('full','compact'):
                    perform(engine,client,state['all'],feature,modality,references[('all',direction,i)],mode)
                for query_number,i0 in enumerate(cohort):
                    i = int(i0); feature = images[i] if modality=='image' else texts[5*i]
                    for repeat in range(args.repeats):
                        modes = ('full','compact') if (repeat+query_number)%2==0 else ('compact','full')
                        for mode in modes:
                            _,_,row = perform(engine,client,state['all'],feature,modality,references[('all',direction,i)],mode)
                            timed.append({'images':size,'direction':direction,'query_number':query_number,
                                'query_cache_image_index':i,'repeat':repeat,'mode_order':modes,**row})
                    print(size,direction,'full/compact query',query_number,'passed',flush=True)
            if size in (100,250):
                for subject in ('one','half'):
                    for direction in ('image-to-text','text-to-image'):
                        modality = 'image' if direction=='image-to-text' else 'text'
                        for query_number,i0 in enumerate(cohort):
                            i = int(i0); feature = images[i] if modality=='image' else texts[5*i]
                            session,response,row = perform(engine,clients[subject],state[subject],feature,modality,
                                references[(subject,direction,i)],'compact')
                            functional.append({'images':size,'permission':subject,'direction':direction,'query_number':query_number,**row})
                            if size==100 and subject=='one' and query_number==0:
                                mutations = tamper_cases(response,engine)
                                bad = copy.deepcopy(response); bad['scores'][0]+=PRIME
                                mutations['prime_modulus_alias_score'] = bad
                                for name,bad in mutations.items():
                                    for variant,flags in VARIANTS.items():
                                        verdict = verify_compact(clients[subject],session,bad,state[subject],**flags)
                                        ablation.append({'direction':direction,'attack':name,'variant':variant,
                                            'accepted':verdict['accepted'],'reason':verdict['reason']})
                                        if variant=='complete':
                                            attacks.append({'direction':direction,'attack':name,'rejected':not verdict['accepted'],
                                                'reason':verdict['reason']})
                                try:
                                    engine.search(clients[subject],session,'compact')
                                except ValueError as error:
                                    attacks.append({'direction':direction,'attack':'query_replay','rejected':True,'reason':str(error)})
                                else:
                                    raise RuntimeError('Replayed compact query accepted')
                                write_json(args.output/f'sample_{direction}_response.json',response)
            for direction in ('image-to-text','text-to-image'):
                i = int(cohort[0]); modality = 'image' if direction=='image-to-text' else 'text'
                feature = images[i] if modality=='image' else texts[5*i]
                _,response,row = perform(engine,clients['none'],state['none'],feature,modality,
                    references[('none',direction,i)],'compact')
                if response['scores'] or response['witnesses'] or response['results']:
                    raise RuntimeError('Zero-rights result not empty')
                functional.append({'images':size,'permission':'none','direction':direction,**row})
        report = {'schema':'ppcmr.private-linear-check.experiment.v1','date':'2026-10-09',
            'features_sha256':provenance['features_sha256'],'manifest_sha256':provenance['manifest_sha256'],
            'runtime':{'python':platform.python_version(),'numpy':np.__version__,
                'cryptography':importlib.metadata.version('cryptography'),'platform':platform.platform(),'BLAS_threads':4},
            'source_sha256':{str(p.relative_to(Path(__file__).parent)):file_sha256(p) for p in
                (Path(__file__),Path(__file__).parent/'ppcmr/compact_verify.py',Path(__file__).parent/'ppcmr/float_mpc.py')},
            'private_state_required':True,'field':'prime 2^521-1','coefficient_PRF':'HMAC-SHA512 with 256-bit private partition seeds',
            'verification':'complete signed compact catalog plus private preprocessed linear score checks plus exact client ranking/content checks',
            'timing':'query construction through final verification/content; cold private-state initialization and owner publication reported separately',
            'distinct_query_pairs':args.queries,'repeats_per_query':args.repeats,'warmup_runs_per_mode_per_scale':2,
            'deployment':deployments,'measurements':timed,'functional':functional,'attacks':attacks,'ablations':ablation,
            'summary':summarize(timed,[r['images'] for r in deployments]),
            'scope':'same-host CPU experiment; no remote deployment, public proof, succinctness, verifier-state disclosure, or collusion guarantee'}
        write_json(args.output/'experiment.json',report)
        if not all(a['rejected'] for a in attacks):
            raise RuntimeError('Complete compact verifier accepted a configured mutation')
        print('completed scale',size,flush=True)
    print(json.dumps({'timed':len(timed),'functional':len(functional),'attacks':len(attacks),
        'rejected':sum(a['rejected'] for a in attacks),'summary':report['summary']},indent=2),flush=True)


if __name__=='__main__':
    main()
