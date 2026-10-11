"""Feature-cached local comparison and measured cold workload prefixes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'

import argparse
import importlib.metadata
import json
import platform
import statistics
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import numpy as np

from float_reference import encode_fp32,exact_dot_encoded
from ppcmr.cached_local import CachedLocalEngine
from ppcmr.cached_mpc import CachedMatrixEngine
from ppcmr.compact_verify import prepare_verifier_state
from ppcmr.crypto import canonical
from ppcmr.float_mpc import TwoCloudEngine,publish
from ppcmr.local_reference import AuthorizedLocalEngine,make_local_query
from research_common import ARTIFACTS,entity_arrays,file_sha256,load_features,write_json
from run_compact_research import perform as perform_mpc
from run_float_mpc_research import build_records
from run_local_comparison import perform as perform_download

PAIRED=('cached_mpc','download_local','cached_local_python','cached_local_limb')
PREFIX=('fresh_mpc','cached_mpc','cached_local_limb')


def perform_cached_local(engine,client,state,feature,modality,reference,method):
    start=perf_counter()
    session=make_local_query(client,feature,modality,10)
    construction=perf_counter()-start
    verdict,cost=engine.search_cached(client,session,state)
    total=perf_counter()-start
    if (not verdict['accepted'] or verdict['scores']!=reference['scores'] or
            verdict['ids']!=reference['ids'] or verdict['contents']!=reference['contents']):
        raise RuntimeError('Cached local retrieval differs from independent exact reference')
    return {'method':method,'query_construction_seconds':construction,'online_total_seconds':total,
        'exact_scores_equal':True,'exact_order_equal':True,'original_contents_equal':True,**cost}


def references_for(records,images,texts,indices):
    out={}
    for direction in ('image-to-text','text-to-image'):
        candidates=[r for r in sorted(records,key=lambda r:r['id']) if
                    r['modality']==('text' if direction=='image-to-text' else 'image')]
        encoded=[encode_fp32(r['embedding']) for r in candidates]
        for i0 in indices:
            i=int(i0);q=encode_fp32(images[i] if direction=='image-to-text' else texts[5*i])
            scores=[exact_dot_encoded(q,row) for row in encoded]
            order=sorted(range(len(candidates)),key=lambda j:(-scores[j],candidates[j]['id']))[:10]
            out[(direction,i)]={'scores':scores,'ids':[candidates[j]['id'] for j in order],
                               'contents':[candidates[j]['content'] for j in order]}
    return out


def summarize_paired(rows):
    result=[]
    for size in sorted({r['images'] for r in rows}):
        for direction in ('image-to-text','text-to-image'):
            groups={m:[r for r in rows if r['images']==size and r['direction']==direction and r['method']==m]
                    for m in PAIRED}
            if not groups[PAIRED[0]]:continue
            pairs={m:{(r['query_number'],r['repeat']):r for r in g} for m,g in groups.items()}
            assert all(pairs[m].keys()==pairs[PAIRED[0]].keys() for m in PAIRED)
            result.append({'images':size,'direction':direction,'candidate_count':groups[PAIRED[0]][0]['candidate_count'],
                **{m:{'runs':len(g),'median_total_ms':statistics.median(r['online_total_seconds'] for r in g)*1000,
                       'min_total_ms':min(r['online_total_seconds'] for r in g)*1000,
                       'max_total_ms':max(r['online_total_seconds'] for r in g)*1000,
                       'median_client_MB':statistics.median(r['client_total_transport_payload_bytes'] for r in g)/1e6,
                       'median_peer_MB':statistics.median(r.get('peer_opening_sent_bytes',0) for r in g)/1e6,
                       'median_dealer_MB':statistics.median(r.get('trusted_preprocessing_sent_bytes',0) for r in g)/1e6}
                   for m,g in groups.items()},
                'median_paired_mpc_over_cached_local':statistics.median(
                    pairs['cached_mpc'][k]['online_total_seconds']/pairs['cached_local_limb'][k]['online_total_seconds']
                    for k in pairs[PAIRED[0]]),
                'median_paired_python_over_limb_local':statistics.median(
                    pairs['cached_local_python'][k]['online_total_seconds']/pairs['cached_local_limb'][k]['online_total_seconds']
                    for k in pairs[PAIRED[0]])})
    return result


def prefix_summary(rows,setup,publication_seconds,queries):
    result=[]
    for q in (1,10,100):
        if q>queries:continue
        prefix=rows[:q]
        query_seconds=sum(r['online_total_seconds'] for r in prefix)
        client=sum(r['client_total_transport_payload_bytes'] for r in prefix)+setup['client_setup_bytes']
        peer=sum(r.get('peer_opening_sent_bytes',0) for r in prefix)+setup['peer_setup_bytes']
        dealer=sum(r.get('trusted_preprocessing_sent_bytes',0) for r in prefix)+setup['dealer_setup_bytes']
        result.append({'queries':q,'query_active_seconds':query_seconds,'cold_setup_seconds':setup['cold_setup_seconds'],
            'setup_plus_query_active_seconds':setup['cold_setup_seconds']+query_seconds,
            'including_common_publication_seconds':publication_seconds+setup['cold_setup_seconds']+query_seconds,
            'client_transport_MB':client/1e6,'peer_transport_MB':peer/1e6,'dealer_transport_MB':dealer/1e6,
            'aggregate_transport_MB':(client+peer+dealer)/1e6,
            'private_state_provisioned_JSON_MB':setup.get('private_state_provisioned_JSON_bytes',0)/1e6})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',type=int,nargs='+',default=[100,250,1000])
    parser.add_argument('--prefix-images',type=int,nargs='+',default=[1000])
    parser.add_argument('--prefix-queries',type=int,default=100)
    parser.add_argument('--phase',choices=['paired','prefix','all'],default='all')
    parser.add_argument('--output',type=Path,default=ARTIFACTS/'float_cached_local')
    args=parser.parse_args()
    if args.output.joinpath('experiment.json').exists():
        raise ValueError('Output already exists; use a new folder to preserve executed evidence')
    manifest,provenance,images,texts=load_features(ARTIFACTS/'test')
    *_,scopes,_=entity_arrays(manifest)
    paired,deployments,workloads=[],[],[]
    source_names=['run_cached_local_research.py','ppcmr/cached_local.py','ppcmr/cached_mpc.py',
        'ppcmr/exact_ring.py','ppcmr/float_mpc.py','ppcmr/compact_verify.py','ppcmr/local_reference.py',
        'ppcmr/crypto.py','float_reference.py','research_common.py','run_compact_research.py',
        'run_local_comparison.py','run_float_mpc_research.py']
    sources={name:file_sha256(Path(__file__).parent/name) for name in source_names}
    report={'schema':'ppcmr.feature-cached-local-and-cold-prefix.v1','date':'2026-10-09',
        'features_sha256':provenance['features_sha256'],'manifest_sha256':provenance['manifest_sha256'],
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,
            'cryptography':importlib.metadata.version('cryptography'),'platform':platform.platform(),
            'BLAS_threads_per_process':4},'source_sha256':sources,
        'timing':'Complete per-query intervals; reference and original-content comparisons outside timing. Cold workload totals sum recorded query-active intervals and separately recorded setup, not uninterrupted batch throughput.',
        'communication':'Serialized IPC payloads excluding framing. Private verifier provisioning JSON size is separate, not actual IPC traffic; common database upload, process-spawn bootstrap and key/credential provisioning are excluded.',
        'scope':'Same-host CPU/private IPC, static signed snapshot, all authorized scopes. Local caches authenticated/decrypted exact features; cached MPC retains matrix masks and private verifier state. No full 5k-image protected run or WAN.',
        'paired_measurements':paired,'deployment':deployments,'cold_workloads':workloads,
        'configured_prefix_queries':args.prefix_queries,'complete':False}
    def save():
        report['paired_summary']=summarize_paired(paired)
        write_json(args.output/'experiment.json',report)
    sizes=sorted(set((args.images if args.phase!='prefix' else [])+
                     (args.prefix_images if args.phase!='paired' else [])))
    for size in sizes:
        if not 100<=size<=len(images) or not 1<=args.prefix_queries<=size:raise ValueError('Invalid size/query count')
        records=build_records(manifest,images,texts,size)
        cohort=np.flatnonzero(scopes[:size]=='scope-00')[:3]
        if len(cohort)!=3:raise ValueError('Not enough paired query features')
        prefix_indices=np.random.default_rng(20261009).permutation(size)[:args.prefix_queries]
        needed=sorted(set(cohort.tolist()+ (prefix_indices.tolist() if size in args.prefix_images and args.phase!='paired' else [])))
        print('precomputing independent reference',size,len(needed),'pairs',flush=True)
        refs=references_for(records,images,texts,needed)
        start=perf_counter()
        server,shares,clients=publish(records,{'all':[f'scope-{i:02d}' for i in range(10)]},
            {'features_sha256':provenance['features_sha256'],'manifest_sha256':provenance['manifest_sha256'],
             'subset_images':size},compact=True)
        publication_seconds=perf_counter()-start;client=clients['all']
        if size in args.images and args.phase!='prefix':
            start=perf_counter();state=prepare_verifier_state(client);state_seconds=perf_counter()-start
            with ExitStack() as stack:
                mpc=stack.enter_context(CachedMatrixEngine(server,shares))
                download=stack.enter_context(AuthorizedLocalEngine(server))
                local=stack.enter_context(CachedLocalEngine(server))
                local_states,local_setup={},{}
                for direction in ('image-to-text','text-to-image'):
                    modality='image' if direction=='image-to-text' else 'text'
                    local_states[direction],local_setup[direction]=local.initialize(client,modality)
                deployments.append({'images':size,'publication_seconds':publication_seconds,
                    'private_state_initialization_seconds':state_seconds,'matrix_setup':mpc.setup_metrics,
                    'local_worker_launch_seconds':local.startup_seconds,'local_cache_initialization':local_setup,
                    'paired_distinct_query_cache_indices':[int(i) for i in cohort]})
                def run(method,feature,modality,direction,i):
                    if method=='cached_mpc':
                        _,_,row=perform_mpc(mpc,client,state,feature,modality,refs[(direction,i)],'compact')
                        return {'method':method,**row}
                    if method=='download_local':return perform_download(method,download,client,feature,modality,refs[(direction,i)])
                    local.kernel='python' if method=='cached_local_python' else 'signed-limb-fp64'
                    return perform_cached_local(local,client,local_states[direction],feature,modality,refs[(direction,i)],method)
                for direction in ('image-to-text','text-to-image'):
                    modality='image' if direction=='image-to-text' else 'text'
                    i=int(cohort[0]);feature=images[i] if modality=='image' else texts[5*i]
                    for method in PAIRED:run(method,feature,modality,direction,i)
                    for number,i0 in enumerate(cohort):
                        i=int(i0);feature=images[i] if modality=='image' else texts[5*i]
                        for repeat in range(3):
                            shift=(number+repeat)%4;order=PAIRED[shift:]+PAIRED[:shift]
                            for method in order:
                                row=run(method,feature,modality,direction,i)
                                paired.append({'images':size,'direction':direction,'query_number':number,
                                    'query_cache_image_index':i,'repeat':repeat,'method_order':order,**row})
                        print('paired',size,direction,number,'four methods exact',flush=True)
            save()
        if size in args.prefix_images and args.phase!='paired':
            for direction in ('image-to-text','text-to-image'):
                modality='image' if direction=='image-to-text' else 'text'
                # Each method starts new workers/cache and immediately runs its stream.
                # Method blocks avoid cross-method initialization delays in cold totals.
                methods=PREFIX if direction=='image-to-text' else tuple(reversed(PREFIX))
                for method in methods:
                    cold_start=perf_counter();private_bytes=0
                    if method!='cached_local_limb':
                        state=prepare_verifier_state(client)
                        private_bytes=len(canonical({f'{a}:{m}':v for (a,m),v in client.verification_sketches.items()}))
                    with ExitStack() as stack:
                        if method=='fresh_mpc':engine=stack.enter_context(TwoCloudEngine(server,shares))
                        elif method=='cached_mpc':engine=stack.enter_context(CachedMatrixEngine(server,shares))
                        else:engine=stack.enter_context(CachedLocalEngine(server))
                        if method=='cached_local_limb':
                            local_state,cache_cost=engine.initialize(client,modality)
                        setup={'cold_setup_seconds':perf_counter()-cold_start,'client_setup_bytes':0,
                               'peer_setup_bytes':0,'dealer_setup_bytes':0,
                               'private_state_provisioned_JSON_bytes':private_bytes}
                        if method=='cached_mpc':
                            setup.update({'matrix_setup':engine.setup_metrics,
                                'peer_setup_bytes':engine.setup_metrics['peer_setup_sent_bytes'],
                                'dealer_setup_bytes':engine.setup_metrics['dealer_setup_sent_bytes'],
                                'control_acknowledgement_bytes':engine.setup_metrics['control_setup_received_bytes']})
                        if method=='cached_local_limb':setup.update({'client_setup_bytes':cache_cost['client_total_transport_payload_bytes'],
                            'local_initialization':cache_cost})
                        stream=[]
                        workload={'images':size,'direction':direction,'method':method,
                            'common_publication_seconds':publication_seconds,'setup':setup,'measurements':stream,
                            'method_block_order':methods,'distinct_query_cache_indices':[int(i) for i in prefix_indices],
                            'prefix_summary':[],'complete':False}
                        workloads.append(workload);save()
                        for number,i0 in enumerate(prefix_indices):
                            i=int(i0);feature=images[i] if modality=='image' else texts[5*i]
                            if method=='cached_local_limb':
                                row=perform_cached_local(engine,client,local_state,feature,modality,refs[(direction,i)],method)
                            else:
                                _,_,row=perform_mpc(engine,client,state,feature,modality,refs[(direction,i)],'compact')
                                row={'method':method,**row}
                            stream.append({'query_number':number,'query_cache_image_index':i,**row})
                            if (number+1)%10==0 or number==0:
                                workload['prefix_summary']=prefix_summary(stream,setup,publication_seconds,len(stream))
                                save();print('cold-prefix',size,direction,method,number+1,'exact queries',flush=True)
                        workload['complete']=True
                        workload['prefix_summary']=prefix_summary(stream,setup,publication_seconds,args.prefix_queries)
                        save()
    assert all(r['complete'] for r in workloads)
    report['complete']=True;save()
    print(json.dumps({'complete':True,'paired_runs':len(paired),'cold_workload_runs':sum(len(w['measurements']) for w in workloads),
                      'output':str(args.output/'experiment.json')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
