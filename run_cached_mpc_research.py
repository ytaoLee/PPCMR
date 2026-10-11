"""Paired fresh/cached MPC, exact-limb kernel ablation, and authorized local search."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'

import argparse
import copy
import importlib.metadata
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np

from float_reference import encode_fp32,exact_dot_encoded
from ppcmr.cached_mpc import CachedMatrixEngine
from ppcmr.compact_verify import PRIME,prepare_verifier_state,verify_compact
from ppcmr.float_mpc import TwoCloudEngine,publish
from ppcmr.local_reference import AuthorizedLocalEngine
from research_common import ARTIFACTS,entity_arrays,file_sha256,load_features,write_json
from run_compact_research import perform
from run_float_mpc_research import build_records,tamper_cases
from run_local_comparison import perform as perform_local

METHODS=('fresh_compact','cached_python','cached_limb','authorized_local')


def summarize(rows,sizes):
    result=[]
    for size in sizes:
        for direction in ('image-to-text','text-to-image'):
            groups={m:[r for r in rows if r['images']==size and r['direction']==direction
                       and r['method']==m] for m in METHODS}
            detail=lambda g:{'runs':len(g),
                'median_total_ms':statistics.median(r['online_total_seconds'] for r in g)*1000,
                'min_total_ms':min(r['online_total_seconds'] for r in g)*1000,
                'max_total_ms':max(r['online_total_seconds'] for r in g)*1000,
                'median_preprocessing_ms':statistics.median(r.get('trusted_preprocessing_seconds',0) for r in g)*1000,
                'median_verify_ms':statistics.median(r.get('verification_seconds',0) for r in g)*1000,
                'median_peer_MB':statistics.median(r.get('peer_opening_sent_bytes',0) for r in g)/1e6,
                'median_dealer_MB':statistics.median(r.get('trusted_preprocessing_sent_bytes',0) for r in g)/1e6,
                'median_client_MB':statistics.median(r['client_total_transport_payload_bytes'] for r in g)/1e6}
            pairs={m:{(r['query_number'],r['repeat']):r for r in g} for m,g in groups.items()}
            assert all(pairs[m].keys()==pairs[METHODS[0]].keys() for m in METHODS)
            ratio=lambda a,b:statistics.median(pairs[a][k]['online_total_seconds']/pairs[b][k]['online_total_seconds']
                                              for k in pairs[a])
            result.append({'images':size,'direction':direction,'candidate_count':groups[METHODS[0]][0]['candidate_count'],
                **{m:detail(g) for m,g in groups.items()},
                'median_paired_fresh_over_cached_python':ratio('fresh_compact','cached_python'),
                'median_paired_cached_python_over_limb':ratio('cached_python','cached_limb'),
                'median_paired_fresh_over_cached_limb':ratio('fresh_compact','cached_limb'),
                'median_paired_cached_limb_over_local':ratio('cached_limb','authorized_local')})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',type=int,nargs='+',default=[100,250,1000])
    parser.add_argument('--queries',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=ARTIFACTS/'float_cached')
    args=parser.parse_args()
    manifest,provenance,images,texts=load_features(ARTIFACTS/'test')
    *_,scopes,_=entity_arrays(manifest)
    timed,functional,attacks,deployments=[],[],[],[]
    for size in args.images:
        if not 10<=size<=len(images) or args.queries<1 or args.repeats<1:raise ValueError('Invalid configuration')
        records=build_records(manifest,images,texts,size)
        cohort=np.flatnonzero(scopes[:size]=='scope-00')[:args.queries]
        if len(cohort)!=args.queries:raise ValueError('Not enough query pairs')
        grants={'none':[],'one':['scope-00'],'half':[f'scope-{j:02d}' for j in range(5)],
                'all':[f'scope-{j:02d}' for j in range(10)]}
        references={}
        for subject,allowed in grants.items():
            for direction in ('image-to-text','text-to-image'):
                candidates=[r for r in sorted(records,key=lambda r:r['id']) if r['scope'] in allowed and
                            r['modality']==('text' if direction=='image-to-text' else 'image')]
                for i0 in cohort:
                    i=int(i0);q=encode_fp32(images[i] if direction=='image-to-text' else texts[5*i])
                    scores=[exact_dot_encoded(q,encode_fp32(r['embedding'])) for r in candidates]
                    order=sorted(range(len(candidates)),key=lambda j:(-scores[j],candidates[j]['id']))[:10]
                    references[(subject,direction,i)]={'scores':scores,'ids':[candidates[j]['id'] for j in order],
                        'contents':[candidates[j]['content'] for j in order]}
        start=perf_counter()
        server,shares,clients=publish(records,grants,{'features_sha256':provenance['features_sha256'],
            'manifest_sha256':provenance['manifest_sha256'],'subset_images':size},compact=True)
        publication_seconds=perf_counter()-start
        states={};state_initialization={}
        for subject,client in clients.items():
            start=perf_counter();states[subject]=prepare_verifier_state(client)
            state_initialization[subject]=perf_counter()-start
        with TwoCloudEngine(server,shares) as fresh, CachedMatrixEngine(server,shares) as cached, AuthorizedLocalEngine(server) as local:
            deployments.append({'images':size,'records':len(records),'publication_seconds':publication_seconds,
                'owner_verification_sketch_seconds':server['publication_metrics']['private_sketch_generation_seconds'],
                'private_state_initialization_seconds':state_initialization,
                'fresh_worker_launch_seconds':fresh.startup_seconds,'cached_worker_launch_seconds':cached.startup_seconds,
                'local_worker_launch_seconds':local.startup_seconds,**cached.setup_metrics})
            def run(method,subject,feature,modality,reference):
                client=clients[subject]
                if method=='authorized_local':return None,None,perform_local(method,local,client,feature,modality,reference)
                engine=fresh if method=='fresh_compact' else cached
                if method!='fresh_compact':cached.kernel='python' if method=='cached_python' else 'limb-fp64'
                session,response,row=perform(engine,client,states[subject],feature,modality,reference,'compact')
                return session,response,{'method':method,**row}
            for direction in ('image-to-text','text-to-image'):
                modality='image' if direction=='image-to-text' else 'text'
                i=int(cohort[0]);feature=images[i] if modality=='image' else texts[5*i]
                for method in METHODS:run(method,'all',feature,modality,references[('all',direction,i)])
                for query_number,i0 in enumerate(cohort):
                    i=int(i0);feature=images[i] if modality=='image' else texts[5*i]
                    for repeat in range(args.repeats):
                        shift=(query_number+repeat)%len(METHODS)
                        order=METHODS[shift:]+METHODS[:shift]
                        for method in order:
                            _,_,row=run(method,'all',feature,modality,references[('all',direction,i)])
                            timed.append({'images':size,'direction':direction,'query_number':query_number,
                                'query_cache_image_index':i,'repeat':repeat,'method_order':order,**row})
                    print(size,direction,'query',query_number,'four modes exact',flush=True)
            if size in (100,250):
                for subject in ('one','half'):
                    for direction in ('image-to-text','text-to-image'):
                        modality='image' if direction=='image-to-text' else 'text'
                        for query_number,i0 in enumerate(cohort):
                            i=int(i0);feature=images[i] if modality=='image' else texts[5*i]
                            session,response,row=run('cached_limb',subject,feature,modality,references[(subject,direction,i)])
                            functional.append({'images':size,'permission':subject,'direction':direction,'query_number':query_number,**row})
                            if size==100 and subject=='one' and query_number==0:
                                cases=tamper_cases(response,cached)
                                bad=copy.deepcopy(response);bad['scores'][0]+=PRIME;cases['prime_modulus_alias_score']=bad
                                for name,bad in cases.items():
                                    verdict=verify_compact(clients[subject],session,bad,states[subject])
                                    attacks.append({'direction':direction,'attack':name,'rejected':not verdict['accepted'],'reason':verdict['reason']})
                                try:cached.search(clients[subject],session,'compact')
                                except ValueError as error:attacks.append({'direction':direction,'attack':'query_replay','rejected':True,'reason':str(error)})
                                else:raise RuntimeError('Replayed cached query accepted')
            for direction in ('image-to-text','text-to-image'):
                i=int(cohort[0]);modality='image' if direction=='image-to-text' else 'text'
                feature=images[i] if modality=='image' else texts[5*i]
                _,_,row=run('cached_limb','none',feature,modality,references[('none',direction,i)])
                functional.append({'images':size,'permission':'none','direction':direction,**row})
        files=['run_cached_mpc_research.py','run_compact_research.py','run_local_comparison.py','run_float_mpc_research.py',
               'ppcmr/cached_mpc.py','ppcmr/exact_ring.py','ppcmr/float_mpc.py','ppcmr/compact_verify.py',
               'ppcmr/local_reference.py','ppcmr/crypto.py','float_reference.py','research_common.py']
        report={'schema':'ppcmr.cached-matrix.experiment.v1','date':'2026-10-09',
            'features_sha256':provenance['features_sha256'],'manifest_sha256':provenance['manifest_sha256'],
            'runtime':{'python':platform.python_version(),'numpy':np.__version__,'cryptography':importlib.metadata.version('cryptography'),
                'platform':platform.platform(),'BLAS_threads_per_process':4},
            'source_sha256':{name:file_sha256(Path(__file__).parent/name) for name in files},
            'distinct_query_pairs':args.queries,'repeats_per_query_per_mode':args.repeats,'methods':METHODS,
            'warmups_per_method_per_scale':2,'matrix_cache_per_immutable_snapshot':True,
            'fresh_query_masks_and_score_shares':True,'no_query_dependent_precision_profile':True,
            'timing':'query construction through final content verification; initial publication, matrix-cache cold setup, worker launch and private-verifier initialization excluded and reported separately; independent reference checks outside timing',
            'scope':'same-host CPU/IPC, exact unchanged 320-bit ring and FP32 values; static snapshot; no WAN or complete 5k-image protected test',
            'deployment':deployments,'measurements':timed,'functional':functional,'attacks':attacks,
            'summary':summarize(timed,[d['images'] for d in deployments])}
        write_json(args.output/'experiment.json',report)
        if not all(a['rejected'] for a in attacks):raise RuntimeError('Cached path accepted a configured attack')
        print('completed',size,flush=True)
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':main()
